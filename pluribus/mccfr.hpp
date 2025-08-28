#pragma once

#include <atomic>
#include <fcntl.h>
#include <filesystem>
#include <libwandb_cpp.h>
#include <memory>
#include <vector>
#include <cereal/cereal.hpp>
#include <cereal/types/polymorphic.hpp>
#include <pluribus/cereal_ext.hpp>
#include <pluribus/config.hpp>
#include <pluribus/decision.hpp>
#include <pluribus/indexing.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/range.hpp>
#include <pluribus/tree_storage.hpp>

namespace pluribus {

int utility(const SlimPokerState& state, int i, const Board& board, const std::vector<Hand>& hands, int stack_size, const RakeStructure& rake,
    const omp::HandEvaluator& eval);

enum class SolverState {
  UNDEFINED, INTERRUPT, SOLVING, SOLVED
};

class Solver : public ConfigProvider {
public:
  explicit Solver(const SolverConfig& config);

  const SolverConfig& get_config() const override { return _config; }
  SolverState get_state() const { return _state; }
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

struct MetricsConfig {
  int max_vpip = 2;
  int max_bet_level = 2;
  std::function<bool(const PokerState&)> should_track = [](const PokerState&) { return true; };
};

template <template<typename> class StorageT>
struct MCCFRContext {
  MCCFRContext(SlimPokerState& state_, const long t_, const int i_, const int consec_folds_, const Board& board_,
      const std::vector<Hand>& hands_, std::vector<CachedIndexer>& indexers_, const omp::HandEvaluator& eval_, StorageT<int>* regret_storage_,
      const StorageT<uint8_t>* bp_node_, SlimPokerState& bp_state_)
    : state{state_}, t{t_}, i{i_}, consec_folds{consec_folds_}, board{board_}, hands{hands_}, indexers{indexers_}, eval{eval_},
      regret_storage{regret_storage_}, bp_node{bp_node_}, bp_state{bp_state_} {}
  MCCFRContext(SlimPokerState& next_state, StorageT<int>* next_regret_storage, const StorageT<uint8_t>* next_bp_node, const int next_consec_folds,
      const MCCFRContext& context)
    : state{next_state}, t{context.t}, i{context.i}, consec_folds{next_consec_folds}, board{context.board}, hands{context.hands},
      indexers{context.indexers}, eval{context.eval}, regret_storage{next_regret_storage}, bp_node{next_bp_node}, bp_state{context.bp_state} {}

  SlimPokerState& state;
  const long t;
  const int i;
  const int consec_folds;
  const Board& board;
  const std::vector<Hand>& hands;
  std::vector<CachedIndexer>& indexers;
  const omp::HandEvaluator& eval;
  StorageT<int>* regret_storage;
  const StorageT<uint8_t>* bp_node;
  SlimPokerState& bp_state;
};

template <template<typename> class StorageT>
struct UpdateContext {
  UpdateContext(SlimPokerState& state_, const int i_, const int consec_folds_, const Board& board_, const std::vector<Hand>& hands_,
      std::vector<CachedIndexer>& indexers_, StorageT<int>* regret_storage_, StorageT<float>* avg_storage_)
    : state{state_}, i{i_}, consec_folds{consec_folds_}, board{board_}, hands{hands_}, indexers{indexers_},
      regret_storage{regret_storage_}, avg_storage{avg_storage_} {}
  UpdateContext(SlimPokerState& next_state, StorageT<int>* next_regret_storage, StorageT<float>* next_avg_storage, const int next_consec_folds,
      const UpdateContext& context)
    : state{next_state}, i{context.i}, consec_folds{next_consec_folds}, board{context.board}, hands{context.hands}, indexers{context.indexers},
      regret_storage{next_regret_storage}, avg_storage{next_avg_storage} {}

  SlimPokerState& state;
  int i;
  int consec_folds;
  const Board& board;
  const std::vector<Hand>& hands;
  std::vector<CachedIndexer>& indexers;
  StorageT<int>* regret_storage;
  StorageT<float>* avg_storage;
};

template <template<typename> class StorageT>
class MCCFRSolver : public Solver {
public:
  explicit MCCFRSolver(const SolverConfig& config) : Solver{config} {}

  long get_iteration() const { return _t; }
  void set_snapshot_dir(const std::string& snapshot_dir) { _snapshot_dir = snapshot_dir; }
  void set_metrics_dir(const std::string& metrics_dir) { _metrics_dir = metrics_dir; }
  void set_log_dir(const std::string& log_dir) { _log_dir = log_dir; }
  void set_regret_metrics_config(const MetricsConfig& metrics_config) { _regret_metrics_config = metrics_config; }
  void interrupt() { _interrupt.store(true, std::memory_order_relaxed); }
  bool is_interrupted() const { return _interrupt.load(std::memory_order_relaxed); }
  virtual void freeze(const std::vector<float>& freq, const Hand& hand, const Board& board, const ActionHistory& history) = 0;

  bool operator==(const MCCFRSolver& other) const { return Solver::operator==(other) && _t == other._t; }

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_t);
  }

protected:
  void _solve(long t_plus) override;
  
  virtual int terminal_utility(const MCCFRContext<StorageT>& context) const;
  virtual bool is_terminal(const SlimPokerState& state, const int i) const { return state.is_terminal() || state.get_players()[i].has_folded(); }
  virtual bool is_frozen(int cluster, const TreeStorageNode<int>* storage) const = 0;
  virtual void on_start() {}
  virtual void on_step(long t, int i, const std::vector<Hand>& hands, std::vector<CachedIndexer>& indexers) {}
  virtual void on_snapshot() {}

  virtual bool should_prune(long t) const = 0;
  virtual bool should_discount(long t) const = 0;
  virtual bool should_snapshot(long t, long T) const = 0;
  virtual bool should_log(long t) const = 0;
  virtual long next_step(long t, long T) const = 0;

  virtual int get_cluster(const SlimPokerState& state, const Board& board, const std::vector<Hand>& hands, std::vector<CachedIndexer>& indexers) = 0;
  virtual std::atomic<int>* get_base_regret_ptr(StorageT<int>* storage, int cluster) = 0;
  virtual std::atomic<float>* get_base_avg_ptr(StorageT<float>* storage, int cluster) = 0;
  virtual StorageT<int>* init_regret_storage() = 0;
  virtual StorageT<float>* init_avg_storage() = 0;
  virtual const StorageT<uint8_t>* init_bp_node() = 0;
  virtual StorageT<int>* next_regret_storage(StorageT<int>* storage, int action_idx, const SlimPokerState& next_state, int i) = 0;
  virtual StorageT<float>* next_avg_storage(StorageT<float>* storage, int action_idx, const SlimPokerState& next_state, int i) = 0;
  virtual const StorageT<uint8_t>* next_bp_node(Action a, const SlimPokerState& state, const StorageT<uint8_t>* bp_node, SlimPokerState& bp_state) = 0;
  virtual const std::vector<Action>& regret_branching_actions(StorageT<int>* storage) const = 0;
  virtual const std::vector<Action>& regret_value_actions(StorageT<int>* storage) const = 0;
  virtual const std::vector<Action>& avg_branching_actions(StorageT<float>* storage) const = 0;
  virtual const std::vector<Action>& avg_value_actions(StorageT<float>* storage) const = 0;
  virtual void save_snapshot(const std::string& fn) const = 0;
  
  virtual double get_discount_factor(long t) const = 0;
  
  virtual void track_regret(nlohmann::json& metrics, std::ostringstream& out_str, long t) const = 0;
  virtual void track_strategy(nlohmann::json& metrics, std::ostringstream& out_str) const = 0;

  std::string track_wandb_metrics(long t) const;
  void track_strategy_by_decision(const PokerState& state, const std::vector<PokerRange>& ranges, const DecisionAlgorithm& decision,
      const MetricsConfig& metrics_config, bool phi, nlohmann::json& metrics) const;

  MetricsConfig get_regret_metrics_config() const { return _regret_metrics_config; }

private:
  int traverse_mccfr_p(const MCCFRContext<StorageT>& ctx);
  int traverse_mccfr(const MCCFRContext<StorageT>& ctx);
  int external_sampling(const std::vector<Action>& actions, const MCCFRContext<StorageT>& ctx);
#ifdef UNIT_TEST
  template <template<typename> class T>
  friend int call_traverse_mccfr(MCCFRSolver<T>* trainer, const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, 
      std::vector<CachedIndexer>& indexers, const omp::HandEvaluator& eval);
#endif

  long _t = 0;
  std::filesystem::path _snapshot_dir = "snapshots";
  std::filesystem::path _metrics_dir = "metrics";
  std::filesystem::path _log_dir = "logs";
  MetricsConfig _regret_metrics_config;
  std::atomic<bool> _interrupt = false;
};

class TreeSolver : virtual public MCCFRSolver<TreeStorageNode>, public Strategy<int> {
public:
  explicit TreeSolver(const SolverConfig& config) : MCCFRSolver{config} {}

  float frequency(Action action, const PokerState& state, const Board& board, const Hand& hand) const override;
  const TreeStorageNode<int>* get_strategy() const override { return _regrets_root.get(); }
  const SolverConfig& get_config() const override { return Solver::get_config(); }

  bool operator==(const TreeSolver& other) const { return MCCFRSolver::operator==(other) && *_regrets_root == *other._regrets_root; }
  
  template <class Archive>
  void serialize(Archive& ar) {
    ar(_regrets_root);
  }

protected:
  void on_start() override;

  std::atomic<int>* get_base_regret_ptr(TreeStorageNode<int>* storage, int cluster) override;
  TreeStorageNode<int>* init_regret_storage() override;
  TreeStorageNode<int>* next_regret_storage(TreeStorageNode<int>* storage, int action_idx, const SlimPokerState& next_state, int i) override;
  const std::vector<Action>& regret_branching_actions(TreeStorageNode<int>* storage) const override;
  const std::vector<Action>& regret_value_actions(TreeStorageNode<int>* storage) const override;
  const std::vector<Action>& avg_branching_actions(TreeStorageNode<float>* storage) const override;
  const std::vector<Action>& avg_value_actions(TreeStorageNode<float>* storage) const override;

  void track_strategy(nlohmann::json& metrics, std::ostringstream& out_str) const override;

  virtual std::shared_ptr<const TreeStorageConfig> make_tree_config() const = 0;

private:
  std::unique_ptr<TreeStorageNode<int>> _regrets_root = nullptr;
};

template <template<typename> class StorageT>
class BlueprintSolver : virtual public MCCFRSolver<StorageT> {
public:
  BlueprintSolver(const SolverConfig& config, const BlueprintSolverConfig& bp_config);

  BlueprintSolverConfig& get_blueprint_config() { return _bp_config; }
  void set_avg_metrics_config(const MetricsConfig& metrics_config) { _avg_metrics_config = metrics_config; }
  bool operator==(const BlueprintSolver& other) const { return MCCFRSolver<StorageT>::operator==(other) && _bp_config == other._bp_config; }

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_bp_config);
  }

protected:
  virtual bool is_update_terminal(const SlimPokerState& state, int i) const;

  void update_strategy(const UpdateContext<StorageT>& ctx);

  void on_start() override;
  void on_step(long t,int i, const std::vector<Hand>& hands, std::vector<CachedIndexer>& indexers) override;

  bool should_prune(long t) const override;
  bool should_discount(const long t) const override { return _bp_config.is_discount_step(t); }
  bool should_snapshot(const long t, const long T) const override { return _bp_config.is_snapshot_step(t, T); }
  bool should_log(const long t) const override { return (t + 1) % _bp_config.log_interval == 0; }
  long next_step(long t, long T) const override;

  int get_cluster(const SlimPokerState& state, const Board& board, const std::vector<Hand>& hands, std::vector<CachedIndexer>& indexers) override;
  double get_discount_factor(const long t) const override { return _bp_config.get_discount_factor(t); }

  MetricsConfig get_avg_metrics_config() const { return _avg_metrics_config; }

  const StorageT<uint8_t>* init_bp_node() override { return nullptr; }
  const StorageT<uint8_t>* next_bp_node(Action a, const SlimPokerState& state, const StorageT<uint8_t>* bp_node, SlimPokerState& bp_state) override { return bp_node; }

private:
  MetricsConfig _avg_metrics_config;
  BlueprintSolverConfig _bp_config;
};

template <template<typename> class StorageT>
class RealTimeSolver : virtual public MCCFRSolver<StorageT> {
public:
  RealTimeSolver(const std::shared_ptr<const SampledBlueprint>& bp, const RealTimeSolverConfig& rt_config);

  const RealTimeSolverConfig& get_real_time_config() const { return _rt_config; }

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_rt_config);
  }

protected:
  int terminal_utility(const MCCFRContext<StorageT>& ctx) const override;
  bool is_terminal(const SlimPokerState& state, int i) const override;

  void on_start() override;
  bool should_prune(long t) const override { return false; /* TODO: test pruning */ }
  bool should_discount(const long t) const override { return t % _rt_config.discount_interval == 0; }
  bool should_snapshot(long t, long T) const override { return false; }
  bool should_log(const long t) const override { return (t + 1) % _rt_config.log_interval == 0; }
  long next_step(const long t, const long T) const override { return std::min(_rt_config.next_discount_step(t, T), t + 20'000'000); }

  int get_cluster(const SlimPokerState& state, const Board& board, const std::vector<Hand>& hands, std::vector<CachedIndexer>& indexers) override;
  double get_discount_factor(const long t) const override { return _rt_config.get_discount_factor(t); }

  std::atomic<float>* get_base_avg_ptr(StorageT<float>* storage, int cluster) override { return nullptr; }
  StorageT<float>* init_avg_storage() override { return nullptr; }
  const StorageT<uint8_t>* init_bp_node() override { return _root_node; }
  StorageT<float>* next_avg_storage(StorageT<float>* storage, int action_idx, const SlimPokerState& next_state, int i) override { return nullptr; }
  const StorageT<uint8_t>* next_bp_node(Action a, const SlimPokerState& state, const StorageT<uint8_t>* bp_node, SlimPokerState& bp_state) override;

private:
  Action next_rollout_action(CachedIndexer& indexer, const SlimPokerState& state, const Hand& hand, const Board& board,
    const TreeStorageNode<uint8_t>* node) const;

  const std::shared_ptr<const SampledBlueprint> _bp = nullptr;
  const TreeStorageNode<uint8_t>* _root_node = nullptr;
  const RealTimeSolverConfig _rt_config;
  const SampledActionProvider _rollout_action_provider;
};

class TreeBlueprintSolver : virtual public TreeSolver, virtual public BlueprintSolver<TreeStorageNode> {
public:
  explicit TreeBlueprintSolver(const SolverConfig& config = SolverConfig{}, const BlueprintSolverConfig& bp_config = BlueprintSolverConfig{});

  const TreeStorageNode<float>* get_phi() const { return _phi_root.get(); }
  void freeze(const std::vector<float>& freq, const Hand& hand, const Board& board, const ActionHistory& history) override;

  bool operator==(const TreeBlueprintSolver& other) const;
  
  template <class Archive>
  void serialize(Archive& ar) {
    ar(_phi_root, cereal::base_class<TreeSolver>(this), cereal::base_class<BlueprintSolver>(this), cereal::base_class<MCCFRSolver>(this),
        cereal::base_class<Solver>(this));
  }

protected:
  void on_start() override;
  void on_snapshot() override;
  bool is_frozen(int cluster, const TreeStorageNode<int>* storage) const override { return false; }

  std::atomic<float>* get_base_avg_ptr(TreeStorageNode<float>* storage, int cluster) override;
  TreeStorageNode<float>* init_avg_storage() override;
  TreeStorageNode<float>* next_avg_storage(TreeStorageNode<float>* storage, int action_idx, const SlimPokerState& next_state, int i) override;
  void save_snapshot(const std::string& fn) const override { cereal_save(*this, fn); }

  void track_regret(nlohmann::json& metrics, std::ostringstream& out_str, long t) const override;
  void track_strategy(nlohmann::json& metrics, std::ostringstream& out_str) const override;

  std::shared_ptr<const TreeStorageConfig> make_tree_config() const override;
  
  const TreeStorageNode<float>* get_phi_root() const { return _phi_root.get(); }

private:
  std::unique_ptr<TreeStorageNode<float>> _phi_root = nullptr;
};

class TreeRealTimeSolver : virtual public TreeSolver, virtual public RealTimeSolver<TreeStorageNode> {
public:
  explicit TreeRealTimeSolver(const SolverConfig& config = SolverConfig{}, const RealTimeSolverConfig& rt_config = RealTimeSolverConfig{},
    const std::shared_ptr<const SampledBlueprint>& bp = nullptr);

  void freeze(const std::vector<float>& freq, const Hand& hand, const Board& board, const ActionHistory& history) override;

  bool operator==(const TreeRealTimeSolver& other) const;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::base_class<TreeSolver>(this), cereal::base_class<RealTimeSolver>(this), cereal::base_class<MCCFRSolver>(this), cereal::base_class<Solver>(this));
  }

protected:
  void on_start() override;
  bool is_frozen(int cluster, const TreeStorageNode<int>* storage) const override;

  void save_snapshot(const std::string& fn) const override { cereal_save(*this, fn); }

  void track_regret(nlohmann::json& metrics, std::ostringstream& out_str, long t) const override {}
  // void track_strategy(nlohmann::json& metrics, std::ostringstream& out_str) const override {}

  std::shared_ptr<const TreeStorageConfig> make_tree_config() const override;
};

}
