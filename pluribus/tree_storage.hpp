#pragma once

#include <atomic>
#include <functional>
#include <mutex>
#include <pluribus/actions.hpp>
#include <pluribus/concurrency.hpp>
#include <pluribus/config.hpp>
#include <pluribus/logging.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/util.hpp>

namespace pluribus {

inline std::vector<Action> real_time_actions(const SlimPokerState& state, const ActionProfile& profile, const RealTimeSolverConfig& rt_config,
    const bool branching) {
  if(state.get_round() >= rt_config.terminal_round || state.get_bet_level() >= rt_config.terminal_bet_level) {
    return branching ? std::vector{{Action::BIAS_DUMMY}} : rt_config.bias_profile.get_actions(state);
  }
  return valid_actions(state, profile);
}

class ActionMode {
public:
  ActionMode() : ActionMode{-1, ActionProfile{}, RealTimeSolverConfig{}, {}} {}

  static ActionMode make_blueprint_mode(const ActionProfile& profile) {
    return ActionMode{0, profile, RealTimeSolverConfig{}, {}};
  }
  static ActionMode make_real_time_mode(const ActionProfile& profile, const RealTimeSolverConfig& rt_config) {
    return ActionMode{1, profile, rt_config, {}};
  }
  static ActionMode make_sampled_mode(const ActionProfile& profile, const std::vector<Action>& biases) {
    return ActionMode{2, profile, RealTimeSolverConfig{}, biases};
  }

  // NOTE: Branching actions must equal value actions except in the bias sampling part of the tree, or in a sampled blueprint.
  //       For bias sampling, branching actions must be a single action, Action::BIAS_DUMMY, to make each player's choice of bias private information.
  //       (i.e. players cannot choose their bias strategy based on knowledge of the biases chosen by previous players)
  //       In a sampled blueprint, value actions are the biases, values are the sampled action, and branching actions are the actions mapping the game tree.
  std::vector<Action> branching_actions(const SlimPokerState& state) const {
    return get_actions(state, true);
  }
  std::vector<Action> value_actions(const SlimPokerState& state) const {
    return get_actions(state, false);
  }

  bool operator==(const ActionMode& other) const = default;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_rt_config, _biases, _profile, _mode);
  }

private:
  std::vector<Action> get_actions(const SlimPokerState& state, const bool branching) const {
    switch(_mode) {
      case 0: return valid_actions(state, _profile);
      case 1: return real_time_actions(state, _profile, _rt_config, branching);
      case 2: return branching ? valid_actions(state, _profile) : _biases;
      default: Logger::error("Unknown action mode: " + std::to_string(_mode));
    }
  }

  ActionMode(const int mode, const ActionProfile& profile,  const RealTimeSolverConfig& rt_config, const std::vector<Action>& biases)
    : _rt_config{rt_config}, _biases{biases}, _profile{profile}, _mode{mode} {}

  std::string to_string() const {
    switch(_mode) {
      case 0: return "Blueprint";
      case 1: return "Real time";
      case 2: return "Sampled blueprint";
      default: Logger::error("Unknown action mode: " + std::to_string(_mode));
    }
  }

  RealTimeSolverConfig _rt_config;
  std::vector<Action> _biases;
  ActionProfile _profile;
  int _mode;
};

class ClusterSpec {
public:
  ClusterSpec(const int preflop , const int flop, const int turn, const int river) : _n_clusters{{preflop, flop, turn, river}} {}
  ClusterSpec() : ClusterSpec{-1, -1, -1, -1} {}

  int n_clusters(const int round) const { return _n_clusters[round]; }

  bool operator==(const ClusterSpec& other) const = default;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_n_clusters);
  }

private:
  std::array<int, 4> _n_clusters;
};

struct TreeStorageConfig {
  ClusterSpec cluster_spec;
  ActionMode action_mode;

  bool operator==(const TreeStorageConfig& other) const = default;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cluster_spec, action_mode);
  }
};

inline int _compute_action_index(const Action a, const std::vector<Action>& actions) {
  const auto it = std::ranges::find(actions, a);
  if(it == actions.end()) Logger::error("Failed to compute action index for action: " + a.to_string());
  return std::distance(actions.begin(), it);
}

inline int node_value_index(const int n_actions, const int cluster, const int action_idx) {
  return n_actions * cluster + action_idx;
}

template <class T>
class TreeStorageNode {
public:
  TreeStorageNode(const SlimPokerState& state, const std::shared_ptr<const TreeStorageConfig>& config) : TreeStorageNode{state, config, true} {
    std::cout << "In constructor.\n";
  }
  TreeStorageNode(): _n_clusters(0), _is_root{true} {}

  ~TreeStorageNode() {
    free_memory();
  }

  TreeStorageNode* apply_index(int action_idx, const SlimPokerState& next_state) {
    auto& node_atom = _nodes[action_idx];
    TreeStorageNode* next = node_atom.load(std::memory_order_acquire);
    if(!next) {
      std::lock_guard lock(_locks[action_idx]);
      next = node_atom.load(std::memory_order_acquire);
      if(!next) {
        next = new TreeStorageNode(next_state, _config, false);
        node_atom.store(next, std::memory_order_release);
      }
    }
    return next;
  }

  const TreeStorageNode* apply_index(int action_idx) const {
    TreeStorageNode* next = _nodes[action_idx].load(std::memory_order_acquire);
    if(!next) Logger::error("TreeStorageNode is not allocated. Index=" + std::to_string(action_idx));
    return next;
  }

  TreeStorageNode* apply(const Action a, const SlimPokerState& next_state) { return apply_index(_compute_action_index(a, _branching_actions), next_state); }
  const TreeStorageNode* apply(const Action a) const { return apply_index(_compute_action_index(a, _branching_actions)); }

  const TreeStorageNode* apply(const std::vector<Action>& actions) const {
    const TreeStorageNode* node = this;
    for(const Action a : actions) {
      node = node->apply(a);
    }
    return node;
  }

  std::atomic<T>* get(const int cluster, const int action_idx = 0) { return &_values[node_value_index(_value_actions.size(), cluster, action_idx)]; }
  const std::atomic<T>* get(const int cluster, const int action_idx = 0) const { return &_values[node_value_index(_value_actions.size(), cluster, action_idx)]; }

  const std::atomic<T>* get_by_index(const int index) const { return &_values[index]; }
  std::atomic<T>* get_by_index(const int index) { return &_values[index]; }

  bool is_allocated(int action_idx) const {
    return _nodes[action_idx].load() != nullptr;
  }

  bool is_allocated(const Action a) const {
    return is_allocated(_compute_action_index(a, _branching_actions));
  }

  const std::vector<Action>& get_branching_actions() const { return _branching_actions; }
  const std::vector<Action>& get_value_actions() const { return _value_actions; }
  int get_n_clusters() const { return _n_clusters; }
  int get_n_values() const { return _value_actions.size() * _n_clusters; }
  std::shared_ptr<const TreeStorageConfig> make_config_ptr() const { return _config; }

  void set_config(const std::shared_ptr<const TreeStorageConfig>& config) {
    _config = config;
    for(int a = 0; a < _branching_actions.size(); ++a) {
      if(TreeStorageNode* child = _nodes[a].load()) {
        child->set_config(config);
      }
    }
  }

  void lcfr_discount(double d) {
    for(int i = 0; i < get_n_values(); ++i) {
      _values[i].store(_values[i].load(std::memory_order_relaxed) * d, std::memory_order_relaxed);
    }
    for(int a = 0; a < _branching_actions.size(); ++a) {
      if(TreeStorageNode* child = _nodes[a].load()) {
        child->lcfr_discount(d);
      }
    }
  }

  bool operator==(const TreeStorageNode& other) const {
    if(_n_clusters != other._n_clusters) return false;
    if(_branching_actions != other._branching_actions) return false;
    if(_value_actions != other._value_actions) return false;
    for(int c = 0; c < _n_clusters; ++c) {
      for(int a = 0; a < _value_actions.size(); ++a) {
        if(get(c, a)->load() != other.get(c, a)->load()) return false;
      }
    }
    for(int a = 0; a < _branching_actions.size(); ++a) {
      TreeStorageNode* lhs = _nodes[a].load();
      TreeStorageNode* rhs = other._nodes[a].load();
      if(lhs && rhs) {
        if(!(*lhs == *rhs)) return false;
      }
      else if(lhs || rhs) {
        return false;
      }
    }
    return true;
  }

  template <class Archive>
  void save(Archive& ar) const {
    ar(_branching_actions, _value_actions, _n_clusters, _is_root);
    if(_is_root) ar(_config);
    for(int c = 0; c < _n_clusters; ++c) {
      for(int a = 0; a < _value_actions.size(); ++a) {
        ar(_values[node_value_index(_value_actions.size(), c, a)]);
      }
    }
    for(int a = 0; a < _branching_actions.size(); ++a) {
      TreeStorageNode* child = _nodes[a].load();
      bool has_child = child != nullptr;
      ar(has_child);
      if(has_child) ar(*child);
    }
  }

  template <class Archive>
  void load(Archive& ar) {
    free_memory();
    ar(_branching_actions, _value_actions, _n_clusters, _is_root);
    if(_is_root) ar(_config);

    _values = std::make_unique<std::atomic<T>[]>(get_n_values());
    _nodes = std::make_unique<std::atomic<TreeStorageNode*>[]>(_branching_actions.size());
    _locks = std::make_unique<SpinLock[]>(_branching_actions.size());

    for(int c = 0; c < _n_clusters; ++c) {
      for(int a = 0; a < _value_actions.size(); ++a) {
        T val;
        ar(val);
        _values[node_value_index(_value_actions.size(), c, a)].store(val);
      }
    }

    for(int a = 0; a < _branching_actions.size(); ++a) {
      bool has_child;
      ar(has_child);
      if(has_child) {
        auto child = new TreeStorageNode();
        ar(*child);
        _nodes[a].store(child);
      }
      else {
        _nodes[a].store(nullptr);
      }
    }

    if(_is_root) set_config(_config);
  }

private:
  TreeStorageNode(const SlimPokerState& state, const std::shared_ptr<const TreeStorageConfig>& config, const bool is_root)
      : _branching_actions{config->action_mode.branching_actions(state)},
        _value_actions{config->action_mode.value_actions(state)},
        _n_clusters{config->cluster_spec.n_clusters(state.get_round())},
        _config{config},
        _values{std::make_unique<std::atomic<T>[]>(get_n_values())},
        _nodes{std::make_unique<std::atomic<TreeStorageNode*>[]>(_branching_actions.size())},
        _locks{std::make_unique<SpinLock[]>(_branching_actions.size())},
        _is_root{is_root} {
    for(int i = 0; i < get_n_values(); ++i) _values[i].store(T{0}, std::memory_order_relaxed);
    for(int i = 0; i < _branching_actions.size(); ++i) _nodes[i].store(nullptr, std::memory_order_relaxed);
  }

  void free_memory() {
    if(!_nodes) return;
    for(int a_idx = 0; a_idx < _branching_actions.size(); ++a_idx) {
      auto& node_atom = _nodes[a_idx];
      if(const TreeStorageNode* node = node_atom.load()) {
        delete node;
        node_atom.store(nullptr);
      }
    }
  }

  std::vector<Action> _branching_actions;
  std::vector<Action> _value_actions;
  int _n_clusters;
  std::shared_ptr<const TreeStorageConfig> _config;

  std::unique_ptr<std::atomic<T>[]> _values;
  std::unique_ptr<std::atomic<TreeStorageNode*>[]> _nodes;
  std::unique_ptr<SpinLock[]> _locks;
  bool _is_root;
};

template<class T>
class Strategy : public ConfigProvider {
public:
  virtual const TreeStorageNode<T>* get_strategy() const = 0;
};

}
