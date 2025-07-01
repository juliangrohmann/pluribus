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
#include <pluribus/tree_storage.hpp>

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
  
  bool operator==(const Solver& other) const { return _config == other._config; }

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_config);
  }

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

template <template<typename> class StorageT>
class MCCFRSolver : public Solver {
public:
  MCCFRSolver(const SolverConfig& config) : Solver{config} {}

  void set_snapshot_dir(std::string snapshot_dir) { _snapshot_dir = snapshot_dir; }
  void set_metrics_dir(std::string metrics_dir) { _metrics_dir = metrics_dir; }
  void set_log_dir(std::string log_dir) { _log_dir = log_dir; }
  void set_log_level(SolverLogLevel log_level) { _log_level = log_level; }

  bool operator==(const MCCFRSolver& other) const { return Solver::operator==(other) && _t == other._t; }

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_t);
  }

protected:
  void _solve(long t_plus) override;
  
  virtual int terminal_utility(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, int stack_size, 
      std::vector<CachedIndexer>& indexers, const omp::HandEvaluator& eval) const { return utility(state, i, board, hands, stack_size, eval); }
  virtual bool is_terminal(const PokerState& state, int i) const { return state.is_terminal() || state.get_players()[i].has_folded(); }
  virtual std::vector<Action> available_actions(const PokerState& state, const ActionProfile& profile) const { return valid_actions(state, profile); }
  virtual void on_start() {}
  virtual void on_step(long t,int i, const std::vector<Hand>& hands, std::ostringstream& debug) {}

  virtual bool should_prune(long t) const = 0;
  virtual bool should_discount(long t) const = 0;
  virtual bool should_snapshot(long t, long T) const = 0;
  virtual bool should_log(long t) const = 0;
  virtual bool is_preflop_frozen(long t) const = 0;
  virtual long next_step(long t, long T) const = 0;
  
  virtual std::atomic<int>* get_base_regret_ptr(StorageT<int>* storage, const PokerState& state, int cluster) = 0;
  virtual std::atomic<float>* get_base_avg_ptr(StorageT<float>* storage, const PokerState& state, int cluster) = 0;
  virtual StorageT<int>* init_regret_storage() = 0;
  virtual StorageT<float>* init_avg_storage() = 0;
  virtual StorageT<int>* next_regret_storage(StorageT<int>* storage, int action_idx, const PokerState& next_state, int i) = 0;
  virtual StorageT<float>* next_avg_storage(StorageT<float>* storage, int action_idx, const PokerState& next_state, int i) = 0;
  virtual std::vector<Action> regret_node_actions(StorageT<int>* storage, const PokerState& state, const ActionProfile& profile) const = 0;
  virtual std::vector<Action> avg_node_actions(StorageT<float>* storage, const PokerState& state, const ActionProfile& profile) const = 0;
  
  virtual double get_discount_factor(long t) const = 0;
  
  virtual void track_regret(nlohmann::json& metrics, std::ostringstream& out_str, long t) const = 0;
  virtual void track_strategy(nlohmann::json& metrics, std::ostringstream& out_str) const = 0;

  std::string track_wandb_metrics(long t) const;
  void track_strategy_by_decision(const DecisionAlgorithm& decision, nlohmann::json& metrics, bool phi) const;

  void error(const std::string& msg, const std::ostringstream& debug) const;

  long get_iteration() const { return _t; }
  SolverLogLevel get_log_level() const { return _log_level; }

private:
  int traverse_mccfr_p(const PokerState& state, long t, int i, const Board& board, const std::vector<Hand>& hands, 
      std::vector<CachedIndexer>& indexers, const omp::HandEvaluator& eval, StorageT<int>* regret_storage, StorageT<float>* avg_storage, 
      std::ostringstream& debug);
  int traverse_mccfr(const PokerState& state, long t, int i, const Board& board, const std::vector<Hand>& hands, 
      std::vector<CachedIndexer>& indexers, const omp::HandEvaluator& eval, StorageT<int>* regret_storage, StorageT<float>* avg_storage,
      std::ostringstream& debug);
  int external_sampling(const std::vector<Action>& actions, const PokerState& state, long t, const Board& board, const std::vector<Hand>& hands, 
      std::vector<CachedIndexer>& indexers, const omp::HandEvaluator& eval, StorageT<int>* regret_storage, StorageT<float>* avg_storage, 
      std::ostringstream& debug);
#ifdef UNIT_TEST
  template <template<typename> class T>
  friend int call_traverse_mccfr(MCCFRSolver<T>* trainer, const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, 
      std::vector<CachedIndexer>& indexers, const omp::HandEvaluator& eval, std::ostringstream& debug);
#endif

  long _t = 0;
  std::filesystem::path _snapshot_dir = "snapshots";
  std::filesystem::path _metrics_dir = "metrics";
  std::filesystem::path _log_dir = "logs";
  SolverLogLevel _log_level = SolverLogLevel::ERRORS;
};

class MappedSolver : virtual public MCCFRSolver<StrategyStorage>, public Strategy<int> {
public:
  MappedSolver(const SolverConfig& config, int n_clusters) : MCCFRSolver{config}, _regrets{config.action_profile, n_clusters} {}

  float frequency(Action action, const PokerState& state, const Board& board, const Hand& hand) const override;
  const StrategyStorage<int>& get_strategy() const override { return _regrets; }
  const SolverConfig& get_config() const override { return Solver::get_config(); }
  
  bool operator==(const MappedSolver& other) const { return MCCFRSolver::operator==(other) && _regrets == other._regrets; }

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_regrets);
  }

protected:
  std::atomic<int>* get_base_regret_ptr(StrategyStorage<int>* storage, const PokerState& state, int cluster) override;
  StrategyStorage<int>* init_regret_storage() override { return &_regrets; }
  StrategyStorage<int>* next_regret_storage(StrategyStorage<int>* storage, int action_idx, const PokerState& next_state, int i) override { return storage; }
  std::vector<Action> regret_node_actions(StrategyStorage<int>* storage, const PokerState& state, const ActionProfile& profile) const override;
  std::vector<Action> avg_node_actions(StrategyStorage<float>* storage, const PokerState& state, const ActionProfile& profile) const override;

  void track_strategy(nlohmann::json& metrics, std::ostringstream& out_str) const override;

private:
  StrategyStorage<int> _regrets;
};

class TreeSolver : virtual public MCCFRSolver<TreeStorageNode> {
public:
  TreeSolver(const SolverConfig& config) : MCCFRSolver{config} {}

  float frequency(Action action, const PokerState& state, const Board& board, const Hand& hand) const;
  const TreeStorageNode<int>* get_regrets_root() const { return _regrets_root.get(); }

  bool operator==(const TreeSolver& other) const { return MCCFRSolver::operator==(other) && *_regrets_root == *other._regrets_root; }
  
  template <class Archive>
  void serialize(Archive& ar) {
    ar(_regrets_root);
    _regrets_root->set_config(make_tree_config());
  }

protected:
  std::atomic<int>* get_base_regret_ptr(TreeStorageNode<int>* storage, const PokerState& state, int cluster) override;
  TreeStorageNode<int>* init_regret_storage() override;
  TreeStorageNode<int>* next_regret_storage(TreeStorageNode<int>* storage, int action_idx, const PokerState& next_state, int i) override;
  std::vector<Action> regret_node_actions(TreeStorageNode<int>* storage, const PokerState& state, const ActionProfile& profile) const override;
  std::vector<Action> avg_node_actions(TreeStorageNode<float>* storage, const PokerState& state, const ActionProfile& profile) const override;

  void track_strategy(nlohmann::json& metrics, std::ostringstream& out_str) const override;

  virtual const std::shared_ptr<const TreeStorageConfig> make_tree_config() const = 0;

  const std::shared_ptr<const TreeStorageConfig> get_regrets_tree_config() { return _regrets_tree_config; }

private:
  std::shared_ptr<const TreeStorageConfig> _regrets_tree_config = nullptr;
  std::unique_ptr<TreeStorageNode<int>> _regrets_root = nullptr;
};

template <template<typename> class StorageT>
class BlueprintSolver : virtual public MCCFRSolver<StorageT> {
public:
  BlueprintSolver(const SolverConfig& config, const BlueprintSolverConfig& bp_config) : _bp_config{bp_config} {}

  const BlueprintSolverConfig& get_blueprint_config() const { return _bp_config; }

  bool operator==(const BlueprintSolver& other) const { return MCCFRSolver<StorageT>::operator==(other) && _bp_config == other._bp_config; }

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_bp_config);
  }

protected:
  void update_strategy(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, StorageT<int>* regret_storage, 
      StorageT<float>* avg_storage, std::ostringstream& debug);
  bool is_update_terminal(const PokerState& state, int i) const;

  void on_start() override { Logger::log("Blueprint solver config:\n" + _bp_config.to_string()); }
  void on_step(long t,int i, const std::vector<Hand>& hands, std::ostringstream& debug) override;

  bool should_prune(long t) const override;
  bool should_discount(long t) const override { return _bp_config.is_discount_step(t); }
  bool should_snapshot(long t, long T) const override { return _bp_config.is_snapshot_step(t, T); }
  bool should_log(long t) const override { return t > 0 && t % _bp_config.log_interval == 0; }
  bool is_preflop_frozen(long t) const override { return false; /* return t > _bp_config.preflop_threshold; */ }
  long next_step(long t, long T) const override;
  
  double get_discount_factor(long t) const override { return _bp_config.get_discount_factor(t); }
  
  #ifdef UNIT_TEST
  template <template<typename> class T>
  friend void call_update_strategy(BlueprintSolver<T>* trainer, const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands,
      StorageT<int>* regret_storage, StorageT<float>* avg_storage, std::ostringstream& debug);
  #endif

private:
  BlueprintSolverConfig _bp_config;
};

template <template<typename> class StorageT>
class RealTimeSolver : virtual public MCCFRSolver<StorageT> {
public:
  RealTimeSolver(const std::shared_ptr<const SampledBlueprint> bp, const RealTimeSolverConfig& rt_config)
      : _bp{bp}, _rt_config{rt_config}, _action_provider{} {}

protected:
  int terminal_utility(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, int stack_size, 
    std::vector<CachedIndexer>& indexers, const omp::HandEvaluator& eval) const override;
  bool is_terminal(const PokerState& state, int i) const override { return state.has_biases() || state.is_terminal() || state.get_players()[i].has_folded(); }
  std::vector<Action> available_actions(const PokerState& state, const ActionProfile& profile) const override;

  void on_start() override { Logger::log("Real time solver config:\n" + _rt_config.to_string()); }
  bool should_prune(long t) const override { return false; /* TODO: test pruning */ }
  bool should_discount(long t) const override { return t % _rt_config.discount_interval == 0; }
  bool should_snapshot(long t, long T) const override { return false; }
  bool should_log(long t) const override { return t % _rt_config.log_interval == 0; }
  bool is_preflop_frozen(long t) const override { return false; }
  long next_step(long t, long T) const override { return _rt_config.next_discount_step(t, T); }
  
  double get_discount_factor(long t) const override { return _rt_config.get_discount_factor(t); }

  std::atomic<float>* get_base_avg_ptr(StrategyStorage<float>* storage, const PokerState& state, int cluster) override { return nullptr; }
  StrategyStorage<float>* init_avg_storage() override { return nullptr; }
  StrategyStorage<float>* next_avg_storage(StrategyStorage<float>* storage, int action_idx, const PokerState& next_state, int i) override { return nullptr; }

private:
  const std::shared_ptr<const SampledBlueprint> _bp = nullptr;
  const RealTimeSolverConfig _rt_config;
  const SampledActionProvider _action_provider;
};

class MappedBlueprintSolver : virtual public MappedSolver, virtual public BlueprintSolver<StrategyStorage> {
public:
  MappedBlueprintSolver(const SolverConfig& config = SolverConfig{}, const BlueprintSolverConfig& bp_config = BlueprintSolverConfig{});

  const StrategyStorage<float>& get_phi() const { return _phi; }

  bool operator==(const MappedBlueprintSolver& other) const;
  
  template <class Archive>
  void serialize(Archive& ar) {
    ar(_phi, cereal::base_class<MappedSolver>(this), cereal::base_class<BlueprintSolver>(this), cereal::base_class<MCCFRSolver>(this), 
        cereal::base_class<Solver>(this));
  }

protected:
  std::atomic<float>* get_base_avg_ptr(StrategyStorage<float>* storage, const PokerState& state, int cluster) override;
  StrategyStorage<float>* init_avg_storage() override { return &_phi; };
  StrategyStorage<float>* next_avg_storage(StrategyStorage<float>* storage, int action_idx, const PokerState& next_state, int i) override { return storage; }

  void track_regret(nlohmann::json& metrics, std::ostringstream& out_str, long t) const override;
  void track_strategy(nlohmann::json& metrics, std::ostringstream& out_str) const override;

private:
  StrategyStorage<float> _phi;
};

class SampledBlueprint;

class MappedRealTimeSolver : virtual public MappedSolver, virtual public RealTimeSolver<StrategyStorage> {
public:
  MappedRealTimeSolver(const std::shared_ptr<const SampledBlueprint> bp, const RealTimeSolverConfig& rt_config = RealTimeSolverConfig{});

protected:
  void track_regret(nlohmann::json& metrics, std::ostringstream& out_str, long t) const override;
};

class TreeBlueprintSolver : virtual public TreeSolver, virtual public BlueprintSolver<TreeStorageNode> {
public:
  TreeBlueprintSolver(const SolverConfig& config = SolverConfig{}, const BlueprintSolverConfig& bp_config = BlueprintSolverConfig{});

  const TreeStorageNode<float>* get_phi() const { return _phi_root.get(); }

  bool operator==(const TreeBlueprintSolver& other) const;
  
  template <class Archive>
  void serialize(Archive& ar) {
    ar(_phi_root, cereal::base_class<TreeSolver>(this), cereal::base_class<BlueprintSolver>(this), cereal::base_class<MCCFRSolver>(this), 
        cereal::base_class<Solver>(this));
    _phi_root->set_config(make_tree_config());
  }

protected:
  std::atomic<float>* get_base_avg_ptr(TreeStorageNode<float>* storage, const PokerState& state, int cluster) override;
  TreeStorageNode<float>* init_avg_storage() override;
  TreeStorageNode<float>* next_avg_storage(TreeStorageNode<float>* storage, int action_idx, const PokerState& next_state, int i) override;

  void track_regret(nlohmann::json& metrics, std::ostringstream& out_str, long t) const override;
  void track_strategy(nlohmann::json& metrics, std::ostringstream& out_str) const override;

  const std::shared_ptr<const TreeStorageConfig> make_tree_config() const override;
  
  const TreeStorageNode<float>* get_phi_root() const { return _phi_root.get(); }
  const std::shared_ptr<const TreeStorageConfig> get_phi_tree_config() { return _phi_tree_config; }

private:
  std::shared_ptr<const TreeStorageConfig> _phi_tree_config = nullptr;
  std::unique_ptr<TreeStorageNode<float>> _phi_root = nullptr;
};

}
