#pragma once

#include <atomic>
#include <mutex>
#include <functional>
#include <pluribus/poker.hpp>
#include <pluribus/actions.hpp>
#include <pluribus/util.hpp>
#include <pluribus/logging.hpp>

namespace pluribus {

struct TreeStorageConfig {
  std::function<int(const PokerState&)> n_clusters_provider;
  std::function<std::vector<Action>(const PokerState&)> action_provider;
};

inline int _compute_action_index(const Action a, const std::vector<Action>& actions) {
  const auto it = std::ranges::find(actions, a);
  // TODO: remove error checking for performance
  if(it == actions.end()) Logger::error("Failed to find action index in TreeStorageNode actions. Action=" + a.to_string()); 
  return std::distance(actions.begin(), it);
}

inline int _value_index(const int n_actions, const int cluster, const int action_idx) {
  return n_actions * cluster + action_idx;
}

template <class T>
class TreeStorageNode {
public:
  TreeStorageNode(const PokerState& state, const std::shared_ptr<const TreeStorageConfig>& config)
      : _actions{config->action_provider(state)}, _n_clusters{config->n_clusters_provider(state)}, _config{std::move(config)},
        _values{std::make_unique<std::atomic<T>[]>(_actions.size() * _n_clusters)}, 
        _nodes{std::make_unique<std::atomic<TreeStorageNode*>[]>(_actions.size())},
        _mutexes{std::make_unique<std::mutex[]>(_actions.size())} {}

  TreeStorageNode(): _n_clusters(0) {}

  ~TreeStorageNode() {
    free_memory();
  }

  TreeStorageNode* apply_index(int action_idx, const PokerState& next_state) {
    auto& node_atom = _nodes[action_idx];
    TreeStorageNode* next = node_atom.load(std::memory_order_seq_cst);
    if(!next) {
      std::lock_guard lock(_mutexes[action_idx]);
      next = node_atom.load(std::memory_order_seq_cst);
      if(!next) {
        next = new TreeStorageNode(next_state, _config);
        node_atom.store(next, std::memory_order_seq_cst);
      }
    }
    return next;
  }

  const TreeStorageNode* apply_index(int action_idx) const {
    TreeStorageNode* next = _nodes[action_idx].load(std::memory_order_seq_cst);
    if(!next) Logger::error("TreeStorageNode is not allocated. Index=" + std::to_string(action_idx));
    return next;
  }

  TreeStorageNode* apply(const Action a, const PokerState& next_state) {
    return apply_index(_compute_action_index(a, _actions), next_state);
  }

  const TreeStorageNode* apply(const Action a) const {
    return apply_index(_compute_action_index(a, _actions));
  }

  std::atomic<T>* get(const int cluster, const int action_idx = 0) {
    return &_values[_value_index(_actions.size(), cluster, action_idx)];
  }

  const std::atomic<T>* get(const int cluster, const int action_idx = 0) const {
    return &_values[_value_index(_actions.size(), cluster, action_idx)];
  }

  bool is_allocated(int action_idx) const {
    return _nodes[action_idx].load() != nullptr;
  }

  bool is_allocated(const Action a) const {
    return is_allocated(_compute_action_index(a, _actions));
  }

  const std::vector<Action>& get_actions() const { return _actions; }

  int get_n_clusters() const { return _n_clusters; }

  void set_config(const std::shared_ptr<const TreeStorageConfig>& config) {
    _config = config;
    for(int a = 0; a < _actions.size(); ++a) {
      if(TreeStorageNode* child = _nodes[a].load()) {
        child->set_config(config);
      }
    }
  }

  void lcfr_discount(double d) {
    for(int i = 0; i < _actions.size() * _n_clusters; ++i) {
      _values[i].store(_values[i].load() * d);
    }
    for(int a = 0; a < _actions.size(); ++a) {
      if(TreeStorageNode* child = _nodes[a].load()) {
        child->lcfr_discount(d);
      }
    }
  }

  bool operator==(const TreeStorageNode& other) const {
    if(_n_clusters != other._n_clusters) return false;
    if(_actions != other._actions) return false;
    for(int c = 0; c < _n_clusters; ++c) {
      for(int a = 0; a < _actions.size(); ++a) {
        if(get(c, a)->load() != other.get(c, a)->load()) return false;
      }
    }
    for(int a = 0; a < _actions.size(); ++a) {
      TreeStorageNode* lhs = _nodes[a].load();
      TreeStorageNode* rhs = other._nodes[a].load();
      if(lhs && rhs) {
        if(!(*lhs == *rhs)) return false;
      } else if(lhs || rhs) {
        return false;
      }
    }
    return true;
  }

  template <class Archive>
  void save(Archive& ar) const {
    ar(_actions, _n_clusters);
    for(int c = 0; c < _n_clusters; ++c) {
      for(int a = 0; a < _actions.size(); ++a) {
        ar(_values[_value_index(_actions.size(), c, a)]);
      }
    }
    for(int a = 0; a < _actions.size(); ++a) {
      TreeStorageNode* child = _nodes[a].load();
      bool has_child = child != nullptr;
      ar(has_child);
      if(has_child) ar(*child);
    }
  }

  template <class Archive>
  void load(Archive& ar) {
    free_memory();
    ar(_actions, _n_clusters);
    int n = _actions.size();
    _values = std::make_unique<std::atomic<T>[]>(n * _n_clusters);
    _nodes = std::make_unique<std::atomic<TreeStorageNode*>[]>(n);
    _mutexes = std::make_unique<std::mutex[]>(n);

    for(int c = 0; c < _n_clusters; ++c) {
      for(int a = 0; a < n; ++a) {
        T val;
        ar(val);
        _values[_value_index(n, c, a)].store(val);
      }
    }

    for(int a = 0; a < n; ++a) {
      bool has_child;
      ar(has_child);
      if(has_child) {
        auto child = new TreeStorageNode();
        ar(*child);
        _nodes[a].store(child);
      }
    }
  }

private:
  void free_memory() {
    if(!_nodes) return;
    for(int a_idx = 0; a_idx < _actions.size(); ++a_idx) {
      auto& node_atom = _nodes[a_idx];
      const TreeStorageNode* node = node_atom.load();
      if(node) {
        delete node;
        node_atom.store(nullptr);
      }
    }
  }

  std::vector<Action> _actions;
  int _n_clusters;
  std::shared_ptr<const TreeStorageConfig> _config;

  std::unique_ptr<std::atomic<T>[]> _values;
  std::unique_ptr<std::atomic<TreeStorageNode*>[]> _nodes;
  std::unique_ptr<std::mutex[]> _mutexes;
};

}