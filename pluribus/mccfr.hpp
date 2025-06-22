#pragma once

#include <array>
#include <vector>
#include <atomic>
#include <memory>
#include <filesystem>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cereal/cereal.hpp>
#include <libwandb_cpp.h>
#include <cereal/types/polymorphic.hpp>
#include <tbb/concurrent_vector.h>
#include <tbb/concurrent_unordered_map.h>
#include <pluribus/range.hpp>
#include <pluribus/cereal_ext.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/indexing.hpp>
#include <pluribus/cluster.hpp>
#include <pluribus/storage.hpp>


namespace pluribus {

template <class T>
std::vector<float> calculate_strategy(const StrategyStorage<T>& data, size_t base_idx, int n_actions) {
  std::vector<float> freq;
  freq.reserve(n_actions);
  float sum = 0;
  for(int a_idx = 0; a_idx < n_actions; ++a_idx) {
    float value = std::max(static_cast<float>(data[base_idx + a_idx].load()), 0.0f);
    freq.push_back(value);
    sum += value;
  }

  if(sum > 0) {
    for(auto& f : freq) {
      f /= sum;
    }
  }
  else {
    float uni = 1.0f / n_actions;
    for(auto& f : freq) {
      f = uni;
    }
  }
  return freq;
}

template <class T>
void lcfr_discount(StrategyStorage<T>* regrets, double d) {
  for(auto& e : regrets->data()) {
    e.store(e.load() * d);
  }
}

template <class T>
std::vector<float> state_to_freq(const PokerState& state, const Board& board, const Hand& hand, 
                            int n_actions, std::vector<CachedIndexer>& indexers, StrategyStorage<T>& strategy) {
  int cluster = FlatClusterMap::get_instance()->cluster(state.get_round(), indexers[state.get_active()].index(board, hand, state.get_round()));
  size_t base_idx = strategy.index(state, cluster);
  return calculate_strategy(strategy, base_idx, n_actions);
}

int sample_action_idx(const std::vector<float>& freq);
int utility(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, int stack_size, const omp::HandEvaluator& eval);

struct MCCFRConfig {
  MCCFRConfig(int n_players = 2, int n_chips = 10'000, int ante = 0);
  MCCFRConfig(const PokerConfig& poker_);

  std::string to_string() const;

  bool operator==(const MCCFRConfig& other) const = default;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(poker, action_profile, init_ranges, init_board, init_state);
  }

  PokerConfig poker;
  ActionProfile action_profile;
  std::vector<PokerRange> init_ranges;
  std::vector<uint8_t> init_board;
  PokerState init_state;
};

struct BlueprintTimingConfig {
  long preflop_threshold_m = 800;
  long snapshot_interval_m = 200;
  long prune_thresh_m = 200;
  long lcfr_thresh_m = 400;
  long discount_interval_m = 10;
  long log_interval_m = 1;
};

struct BlueprintTrainerConfig {
  BlueprintTrainerConfig();

  std::string to_string() const;

  void set_iterations(const BlueprintTimingConfig& timings, long it_per_min);
  long next_discount_step(long t, long T) const;
  long next_snapshot_step(long t, long T) const;
  bool is_discount_step(long t) const;
  bool is_snapshot_step(long t, long T) const;

  bool operator==(const BlueprintTrainerConfig& other) const = default;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(strategy_interval, preflop_threshold, snapshot_interval, prune_thresh, lcfr_thresh, discount_interval, log_interval);
  }

  long strategy_interval = 10'000;
  long preflop_threshold;
  long snapshot_interval;
  long prune_thresh;
  long lcfr_thresh;
  long discount_interval;
  long log_interval;
};

template<class T>
class Strategy {
public:
  virtual const StrategyStorage<T>& get_strategy() const = 0;
  virtual const MCCFRConfig& get_config() const = 0;
};

enum class BlueprintLogLevel : int {
  NONE = 0,
  ERRORS = 1,
  DEBUG = 2
};

class MCCFRTrainer : public Strategy<int> {
public:
  MCCFRTrainer(const MCCFRConfig& mccfr_config);
  
  void mccfr_p(long t_plus);
  void allocate_all();

  const MCCFRConfig& get_config() const { return _mccfr_config; }
  void set_snapshot_dir(std::string snapshot_dir) { _snapshot_dir = snapshot_dir; }
  void set_metrics_dir(std::string metrics_dir) { _metrics_dir = metrics_dir; }
  void set_log_dir(std::string log_dir) { _log_dir = log_dir; }
  void set_log_level(BlueprintLogLevel log_level) { _log_level = log_level; }

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_t);
  }

  virtual ~MCCFRTrainer();
  
protected:
  virtual void on_start() {}
  virtual void on_step(long t,int i, const std::vector<Hand>& hands, std::ostringstream& debug) {}

  virtual bool should_prune(long t) const = 0;
  virtual bool should_discount(long t) const = 0;
  virtual bool should_snapshot(long t, long T) const = 0;
  virtual bool should_log(long t) const = 0;
  virtual bool is_preflop_frozen(long t) const = 0;
  virtual long next_step(long t, long T) const = 0;
  
  virtual StrategyStorage<int>* get_regrets() = 0;
  virtual StrategyStorage<float>* get_avg_strategy() = 0;

  virtual double get_discount_factor(long t) const = 0;

  virtual std::string build_wandb_metrics(long t) const = 0;
  void error(const std::string& msg, const std::ostringstream& debug) const;

  long get_iteration() const { return _t; }
  BlueprintLogLevel get_log_level() const { return _log_level; }

private:
  int traverse_mccfr_p(const PokerState& state, long t, int i, const Board& board, const std::vector<Hand>& hands, 
      std::vector<CachedIndexer>& indexers, const omp::HandEvaluator& eval, std::ostringstream& debug);
  int traverse_mccfr(const PokerState& state, long t, int i, const Board& board, const std::vector<Hand>& hands, 
      std::vector<CachedIndexer>& indexers, const omp::HandEvaluator& eval, std::ostringstream& debug);
  void _allocate_state(const PokerState& state);

  void log_utility(int utility, const PokerState& state, const std::vector<Hand>& hands, std::ostringstream& debug) const;
  void log_action_ev(Action a, float freq, int ev, const PokerState& state, std::ostringstream& debug) const;
  void log_net_ev(int ev, float ev_exact, const PokerState& state, std::ostringstream& debug) const;
  void log_regret(Action a, int d_r, int total_r, std::ostringstream& debug) const;
  void log_external_sampling(Action sampled, const std::vector<Action>& actions, const std::vector<float>& freq,
                             const PokerState& state, std::ostringstream& debug) const;
#ifdef UNIT_TEST
  friend int call_traverse_mccfr(MCCFRTrainer* trainer, const PokerState& state, int i, const Board& board, 
      const std::vector<Hand>& hands, std::vector<CachedIndexer>& indexers, const omp::HandEvaluator& eval, std::ostringstream& debug);
#endif

  MCCFRConfig _mccfr_config;
  long _t = 1;
  std::filesystem::path _snapshot_dir = "snapshots";
  std::filesystem::path _metrics_dir = "metrics";
  std::filesystem::path _log_dir = "logs";
  BlueprintLogLevel _log_level = BlueprintLogLevel::ERRORS;
};

class BlueprintTrainer : public MCCFRTrainer {
public:
  BlueprintTrainer(const BlueprintTrainerConfig& bp_config = BlueprintTrainerConfig{}, const MCCFRConfig& mccfr_config = MCCFRConfig{});
  bool operator==(const BlueprintTrainer& other) const;
  const StrategyStorage<int>& get_strategy() const { return _regrets; }
  const StrategyStorage<float>& get_phi() const { return _phi; }
  const BlueprintTrainerConfig& get_blueprint_config() const { return _bp_config; }

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_regrets, _phi, _bp_config, cereal::base_class<MCCFRTrainer>(this));
  }

protected:
  void on_start() override;
  void on_step(long t,int i, const std::vector<Hand>& hands, std::ostringstream& debug) override;

  bool should_prune(long t) const override;
  bool should_discount(long t) const override;
  bool should_snapshot(long t, long T) const override;
  bool should_log(long t) const override;
  bool is_preflop_frozen(long t) const override;
  long next_step(long t, long T) const override;

  StrategyStorage<int>* get_regrets() override { return &_regrets; }
  StrategyStorage<float>* get_avg_strategy() override { return &_phi; }

  double get_discount_factor(long t) const override;
  std::string build_wandb_metrics(long t) const override;

private:
  void update_strategy(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, std::ostringstream& debug);

#ifdef UNIT_TEST
  friend void call_update_strategy(BlueprintTrainer* trainer, const PokerState& state, int i, const Board& board,
                                   const std::vector<Hand>& hands, std::ostringstream& debug);
#endif

  StrategyStorage<int> _regrets;
  StrategyStorage<float> _phi;
  BlueprintTrainerConfig _bp_config;
};

}
