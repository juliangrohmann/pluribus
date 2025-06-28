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

int sample_action_idx(const std::vector<float>& freq);
int utility(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, int stack_size, const omp::HandEvaluator& eval);

struct SolverConfig {
  SolverConfig(int n_players = 2, int n_chips = 10'000, int ante = 0);
  SolverConfig(const PokerConfig& poker_);

  std::string to_string() const;

  bool operator==(const SolverConfig& other) const = default;

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

enum class SolverState {
  UNDEFINED, INTERRUPT, SOLVING, SOLVED
};

class ConfigProvider {
public:
  virtual const SolverConfig& get_config() const = 0;
  virtual ~ConfigProvider() = default;
};

class Solver : public ConfigProvider {
public:
  Solver(const SolverConfig& config);

  const SolverConfig& get_config() const override { return _config; }
  SolverState get_state() { return _state; }; 
  void solve(long t_plus);
  virtual float frequency(Action action, const PokerState& state, const Board& board, const Hand& hand) const = 0;

protected:
  virtual void _solve(long t_plus) = 0;
  
private:
  SolverState _state = SolverState::UNDEFINED;
  SolverConfig _config;
};

template<class T>
class Strategy : public ConfigProvider {
public:
  virtual const StrategyStorage<T>& get_strategy() const = 0;
};

enum class SolverLogLevel : int {
  NONE = 0,
  ERRORS = 1,
  DEBUG = 2
};

class MCCFRSolver : public Strategy<int>, public Solver {
public:
  MCCFRSolver(const SolverConfig& config);

  void allocate_all();

  float frequency(Action action, const PokerState& state, const Board& board, const Hand& hand) const;
  const SolverConfig& get_config() const override { return Solver::get_config(); }
  void set_snapshot_dir(std::string snapshot_dir) { _snapshot_dir = snapshot_dir; }
  void set_metrics_dir(std::string metrics_dir) { _metrics_dir = metrics_dir; }
  void set_log_dir(std::string log_dir) { _log_dir = log_dir; }
  void set_log_level(SolverLogLevel log_level) { _log_level = log_level; }

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_t);
  }

  virtual ~MCCFRSolver();
  
protected:
  void _solve(long t_plus) override;

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
  SolverLogLevel get_log_level() const { return _log_level; }

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
  friend int call_traverse_mccfr(MCCFRSolver* trainer, const PokerState& state, int i, const Board& board, 
      const std::vector<Hand>& hands, std::vector<CachedIndexer>& indexers, const omp::HandEvaluator& eval, std::ostringstream& debug);
#endif

  long _t = 0;
  std::filesystem::path _snapshot_dir = "snapshots";
  std::filesystem::path _metrics_dir = "metrics";
  std::filesystem::path _log_dir = "logs";
  SolverLogLevel _log_level = SolverLogLevel::ERRORS;
};

struct BlueprintTimingConfig {
  long preflop_threshold_m = 800;
  long snapshot_interval_m = 200;
  long prune_thresh_m = 200;
  long lcfr_thresh_m = 400;
  long discount_interval_m = 10;
  long log_interval_m = 1;
};

struct BlueprintSolverConfig {
  BlueprintSolverConfig();

  std::string to_string() const;

  void set_iterations(const BlueprintTimingConfig& timings, long it_per_min);
  long next_discount_step(long t, long T) const;
  long next_snapshot_step(long t, long T) const;
  bool is_discount_step(long t) const;
  bool is_snapshot_step(long t, long T) const;

  bool operator==(const BlueprintSolverConfig& other) const = default;

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

class BlueprintSolver : public MCCFRSolver {
public:
  BlueprintSolver(const BlueprintSolverConfig& bp_config = BlueprintSolverConfig{}, const SolverConfig& mccfr_config = SolverConfig{});
  bool operator==(const BlueprintSolver& other) const;
  const StrategyStorage<int>& get_strategy() const { return _regrets; }
  const StrategyStorage<float>& get_phi() const { return _phi; }
  const BlueprintSolverConfig& get_blueprint_config() const { return _bp_config; }

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_regrets, _phi, _bp_config, cereal::base_class<MCCFRSolver>(this));
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
  friend void call_update_strategy(BlueprintSolver* trainer, const PokerState& state, int i, const Board& board,
                                   const std::vector<Hand>& hands, std::ostringstream& debug);
#endif

  StrategyStorage<int> _regrets;
  StrategyStorage<float> _phi;
  BlueprintSolverConfig _bp_config;
};

class SampledBlueprint;

class RealTimeMCCFR : public Solver {
public:
  RealTimeMCCFR(const SolverConfig& config, const std::shared_ptr<const SampledBlueprint> bp) : Solver{config}, _bp{bp} {}
  float frequency(Action action, const PokerState& state, const Board& board, const Hand& hand) const override;

protected:
  void _solve(long t_plus) override;

private:
  const std::shared_ptr<const SampledBlueprint> _bp = nullptr;
  std::unique_ptr<StrategyStorage<int>> _regrets = nullptr;
};

}
