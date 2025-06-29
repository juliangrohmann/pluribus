#pragma once

#include <atomic>
#include <mutex>
#include <functional>
#include <pluribus/poker.hpp>
#include <pluribus/actions.hpp>

namespace pluribus {

struct MCCFRTreeConfig {
  std::function<std::vector<Action>(const PokerState&)> action_provider;
};

template<class T>
class MCCFRNode {
public:
  MCCFRNode(const std::vector<Action>& actions, int n_clusters, const MCCFRTreeConfig* config) 
      : _actions{actions}, _n_clusters{n_clusters}, _config{config} {
    _values = new std::atomic<T>[actions.size() * n_clusters];
    _nodes = new std::atomic<MCCFRNode*>[actions.size()]{};
    _mutexes = new std::mutex[actions.size()];
    // _actions = new Action[actions.size()];
    // for(int i = 0; i < actions.size(); ++i) _actions[i] = actions[i];
  }

  MCCFRNode* apply_index(int action_idx, int n_clusters, const PokerState& next_state) {
    std::atomic<MCCFRNode*>& node_atom = _nodes[action_idx];
    MCCFRNode* next = node_atom.load(std::memory_order_seq_cst); // std::memory_order_acquire for performance
    if(!next) {
      std::lock_guard<std::mutex> lock(_mutexes[action_idx]); 
      next = node_atom.load(std::memory_order_seq_cst); // std::memory_order_relaxed for performance
      if(!next) {
        next = new MCCFRNode(config->action_provider(next_state), n_clusters, config);
        node_atom.store(next, std::memory_order_seq_cst); // std::memory_order_release for performance
      }
    }
    return next;
  }

  MCCFRNode* apply(Action a, int n_clusters, const PokerState& next_state) {
    int a_idx = std::distance(actions.begin(), std::find(actions.begin(), actions.end(), a));
    return apply_index(a_idx, n_clusters, next_state);
  }

  std::atomic<T> get(int n_actions, int cluster, int action_idx = 0) {
    return _values[n_actions * cluster + action_idx];
  }
  
private: 
  std::atomic<T>* _values;
  std::atomic<MCCFRNode*>* _nodes;
  std::mutex* _mutexes;
  std::vector<Action> _actions;
  const MCCFRTreeConfig* config;
  // Action* _actions;
  int _n_clusters;
  // uint8_t _n_actions;
};

}
