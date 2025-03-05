#pragma once

#include <atomic>
#include <iostream>
#include <filesystem>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_vector.h>
#include <omp.h>
#include <pluribus/cereal_ext.hpp>
#include <pluribus/infoset.hpp>
#include <pluribus/history_index.hpp>
#include <pluribus/actions.hpp>

namespace pluribus {

template<class T>
class StrategyStorage {
public:
  StrategyStorage(const ActionProfile& action_profile = BlueprintActionProfile{}, int n_clusters = 200) : 
               _action_profile{action_profile}, _n_clusters{n_clusters} {};

  inline const tbb::concurrent_vector<std::atomic<T>>& data() const { return _data; }
  inline tbb::concurrent_vector<std::atomic<T>>& data() { return _data; }
  inline const tbb::concurrent_unordered_map<ActionHistory, int> history_map() const { return _history_map; }
  inline T get_n_clusters() const { return _n_clusters; }

  std::atomic<T>& operator[](size_t idx) { return _data[idx]; }
  const std::atomic<T>& operator[](size_t idx) const { return _data[idx]; }

  size_t index(const PokerState& state, int cluster, int action = 0) {
    auto actions = valid_actions(state, _action_profile);
    size_t n_actions = actions.size();
    auto history = state.get_action_history();

    size_t history_idx;
    auto it = _history_map.find(history);
    if (it != _history_map.end()) {
      history_idx = it->second;
    } else {
      std::lock_guard<std::mutex> lock(_grow_mutex);
      it = _history_map.find(history); // Double-check
      if (it == _history_map.end()) {
        history_idx = _data.size();
        _history_map.insert({history, history_idx});
        _data.grow_by(_n_clusters * n_actions);
      } else {
        history_idx = it->second;
      }
    }

    // Ensure growth is complete before returning
    size_t max_idx = history_idx + _n_clusters * n_actions - 1;
    while (max_idx >= _data.size()) {
      std::this_thread::yield(); // Wait for growth if another thread is still growing
    }
    (void)_data[max_idx].load(std::memory_order_acquire); // Ensure construction is visible

    return history_idx + cluster * n_actions + action;
  }

  size_t index(const PokerState& state, int cluster, int action = 0) const {
    size_t n_actions = valid_actions(state, _action_profile).size();
    auto it = _history_map.find(state.get_action_history());
    if(it != _history_map.end()) return it->second + cluster * n_actions + action;
    throw std::runtime_error("StrategyStorage --- Indexed out of range.");
  }

  bool operator==(const StrategyStorage& other) const {
    return _data == other._data &&
           _history_map == other._history_map &&
           _action_profile == other._action_profile &&
           _n_clusters == other._n_clusters;
  }

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_data, _history_map, _action_profile, _n_clusters);
  }

private:
  tbb::concurrent_vector<std::atomic<T>> _data;
  tbb::concurrent_unordered_map<ActionHistory, int> _history_map;
  ActionProfile _action_profile;
  int _n_clusters;
  std::mutex _grow_mutex;
};

}