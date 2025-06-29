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
#include <pluribus/decision.hpp>
#include <pluribus/config.hpp>
#include <pluribus/storage.hpp>

namespace pluribus {

int utility(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, int stack_size, const omp::HandEvaluator& eval);

enum class SolverState {
  UNDEFINED, INTERRUPT, SOLVING, SOLVED
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

enum class SolverLogLevel : int {
  NONE = 0,
  ERRORS = 1,
  DEBUG = 2
};

class MCCFRSolver : public Strategy<int>, public Solver {
public:
  MCCFRSolver(const SolverConfig& config) : Solver{config} {}

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

  virtual ~MCCFRSolver() = default;
  
protected:
  void _solve(long t_plus) override;
  
  virtual int terminal_utility(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, int stack_size, 
      std::vector<CachedIndexer>& indexers, const omp::HandEvaluator& eval) const { return utility(state, i, board, hands, stack_size, eval); }
  virtual bool is_terminal(const PokerState& state) const { return state.is_terminal(); }
  virtual std::vector<Action> available_actions(const PokerState& state, const ActionProfile& profile) const { return valid_actions(state, profile); }
  
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
  
  template <class T>
  void log_strategy(const StrategyStorage<T>& strat, const SolverConfig& config, nlohmann::json& metrics, bool phi) const;
  
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
  void on_start() override { Logger::log("Blueprint solver config:\n" + _bp_config.to_string()); }
  void on_step(long t,int i, const std::vector<Hand>& hands, std::ostringstream& debug) override;

  bool should_prune(long t) const override;
  bool should_discount(long t) const override { return _bp_config.is_discount_step(t); }
  bool should_snapshot(long t, long T) const override { return _bp_config.is_snapshot_step(t, T); }
  bool should_log(long t) const override { return t > 0 && t % _bp_config.log_interval == 0; }
  bool is_preflop_frozen(long t) const override { return t > _bp_config.preflop_threshold; }
  long next_step(long t, long T) const override;

  StrategyStorage<int>* get_regrets() override { return &_regrets; }
  StrategyStorage<float>* get_avg_strategy() override { return &_phi; }

  double get_discount_factor(long t) const override { return _bp_config.get_discount_factor(t); }
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

class RealTimeMCCFR : public MCCFRSolver {
public:
  RealTimeMCCFR(const SolverConfig& config, RealTimeSolverConfig rt_config, const std::shared_ptr<const SampledBlueprint> bp);
  const StrategyStorage<int>& get_strategy() const { return _regrets; }

protected:

  int terminal_utility(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, int stack_size, 
    std::vector<CachedIndexer>& indexers, const omp::HandEvaluator& eval) const override;
  bool is_terminal(const PokerState& state) const override;
  std::vector<Action> available_actions(const PokerState& state, const ActionProfile& profile) const override;

  void on_start() override { Logger::log("Real time solver config:\n" + _rt_config.to_string()); }
  bool should_prune(long t) const override { return false; /* TODO: test pruning */ }
  bool should_discount(long t) const override { return t % _rt_config.discount_interval == 0; }
  bool should_snapshot(long t, long T) const override { return false; }
  bool should_log(long t) const override { return t % _rt_config.log_interval == 0; }
  bool is_preflop_frozen(long t) const override { return false; }
  long next_step(long t, long T) const override { return _rt_config.next_discount_step(t, T); }
  
  StrategyStorage<int>* get_regrets() override { return &_regrets; };
  StrategyStorage<float>* get_avg_strategy() override { return nullptr; };

  double get_discount_factor(long t) const override { return _rt_config.get_discount_factor(t); }

  std::string build_wandb_metrics(long t) const override;

private:
  const std::shared_ptr<const SampledBlueprint> _bp = nullptr;
  StrategyStorage<int> _regrets;
  RealTimeSolverConfig _rt_config;
  SampledActionProvider _action_provider;
};

}
