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
#include <tbb/concurrent_vector.h>
#include <tbb/concurrent_unordered_map.h>
#include <pluribus/range.hpp>
#include <pluribus/cereal_ext.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/infoset.hpp>
#include <pluribus/cluster.hpp>
#include <pluribus/storage.hpp>


namespace pluribus {

// using PreflopMap = tbb::concurrent_unordered_map<InformationSet, tbb::concurrent_vector<float>>;

template <class T>
std::vector<float> calculate_strategy(const StrategyStorage<T>& data, size_t base_idx, int n_actions) {
  T sum = 0;
  for(int a_idx = 0; a_idx < n_actions; ++a_idx) {
    sum += std::max(data[base_idx + a_idx].load(), static_cast<T>(0));
  }

  std::vector<float> freq;
  freq.reserve(n_actions);
  if(sum > 0) {
    for(int a_idx = 0; a_idx < n_actions; ++a_idx) {
      freq.push_back(std::max(data[base_idx + a_idx].load(), static_cast<T>(0)) / static_cast<double>(sum));
    }
  }
  else {
    for(int i = 0; i < n_actions; ++i) {
      freq.push_back(1.0 / n_actions);
    }
  }
  return freq;
}

template <class T>
void lcfr_discount(StrategyStorage<T>& regrets, double d) {
  for(auto& e : regrets.data()) {
    e.store(e.load() * d);
  }
}

template <class T>
std::vector<float> get_freq(const PokerState& state, const Board& board, const Hand& hand, 
                            int n_actions, StrategyStorage<T>& strategy) {
  int cluster = FlatClusterMap::get_instance()->cluster(state.get_round(), board, hand);
  size_t base_idx = strategy.index(state, cluster);
  return calculate_strategy(strategy, base_idx, n_actions);
}

int sample_action_idx(const std::vector<float>& freq);
int utility(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, int stack_size, const omp::HandEvaluator& eval);

struct BlueprintTimingConfig {
  long preflop_threshold_m = 800;
  long snapshot_interval_m = 200;
  long prune_thresh_m = 200;
  long lcfr_thresh_m = 400;
  long discount_interval_m = 10;
  long log_interval_m = 1;
};

struct BlueprintTrainerConfig {
  BlueprintTrainerConfig(int n_players = 2, int n_chips = 10'000, int ante = 0);
  BlueprintTrainerConfig(const PokerConfig& poker_);

  std::string to_string() const;

  bool operator==(const BlueprintTrainerConfig&) const = default;

  void set_iterations(const BlueprintTimingConfig& timings, long it_per_min);
  long next_discount_step(long t, long T) const;
  long next_snapshot_step(long t, long T) const;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(poker, action_profile, init_ranges, init_board, init_state, strategy_interval, preflop_threshold, snapshot_interval, 
       prune_thresh, lcfr_thresh, discount_interval, log_interval, prune_cutoff, regret_floor);
  }

  PokerConfig poker;
  ActionProfile action_profile;
  std::vector<PokerRange> init_ranges;
  std::vector<uint8_t> init_board;
  PokerState init_state;
  long strategy_interval = 10'000;
  long preflop_threshold;
  long snapshot_interval;
  long prune_thresh;
  long lcfr_thresh;
  long discount_interval;
  long log_interval;
  int prune_cutoff = -300'000'000;
  int regret_floor = -310'000'000;
};

template<class T>
class Strategy {
public:
  virtual const StrategyStorage<T>& get_strategy() const = 0;
  virtual const BlueprintTrainerConfig& get_config() const = 0;
};

enum class BlueprintLogLevel : int {
  NONE = 0,
  ERRORS = 1,
  DEBUG = 2
};

class BlueprintTrainer : public Strategy<int> {
public:
  BlueprintTrainer(const BlueprintTrainerConfig& config = BlueprintTrainerConfig{}, bool enable_wandb = false);
  void mccfr_p(long t_plus);
  bool operator==(const BlueprintTrainer& other) const;
  const StrategyStorage<int>& get_strategy() const { return _regrets; }
  StrategyStorage<int>& get_strategy() { return _regrets; }
  const StrategyStorage<float>& get_phi() const { return _phi; }
  const BlueprintTrainerConfig& get_config() const { return _config; }
  BlueprintTrainerConfig& get_config() { return _config; }
  void set_snapshot_dir(std::string snapshot_dir) { _snapshot_dir = snapshot_dir; }
  void set_metrics_dir(std::string metrics_dir) { _metrics_dir = metrics_dir; }
  void set_log_dir(std::string log_dir) { _log_dir = log_dir; }
  void set_log_level(BlueprintLogLevel log_level) { _log_level = log_level; }

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_regrets, _phi, _config, _t);
  }

private:
  int traverse_mccfr_p(const PokerState& state, long t, int i, const Board& board, const std::vector<Hand>& hands, const omp::HandEvaluator& eval, std::ostringstream& debug);
  int traverse_mccfr(const PokerState& state, long t, int i, const Board& board, const std::vector<Hand>& hands, const omp::HandEvaluator& eval, std::ostringstream& debug);
  void update_strategy(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, std::ostringstream& debug);
  void log_metrics(long t);
  void error(const std::string& msg, const std::ostringstream& debug) const;

#ifdef UNIT_TEST
  friend int call_traverse_mccfr(BlueprintTrainer& trainer, const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, 
                                 const omp::HandEvaluator& eval, std::ostringstream& debug);
  friend void call_update_strategy(BlueprintTrainer& trainer, const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands,
                                   std::ostringstream& debug);
#endif
  StrategyStorage<int> _regrets;
  StrategyStorage<float> _phi;
  BlueprintTrainerConfig _config;
  std::filesystem::path _snapshot_dir = "snapshots";
  std::filesystem::path _metrics_dir = "metrics";
  std::filesystem::path _log_dir = "logs";
  std::unique_ptr<wandb::Session> _wb;
  wandb::Run _wb_run;
  long _t;
  BlueprintLogLevel _log_level = BlueprintLogLevel::NONE;
};

}
