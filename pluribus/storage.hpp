#pragma once

#include <atomic>
#include <iostream>
#include <filesystem>
#include <mutex>
#include <condition_variable>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_vector.h>
#include <omp.h>
#include <pluribus/cereal_ext.hpp>
#include <pluribus/indexing.hpp>
#include <pluribus/history_index.hpp>
#include <pluribus/actions.hpp>

namespace pluribus {

struct HistoryEntry {
  HistoryEntry(size_t i = 0, bool init = false) : idx{i}, ready{init} {}
  HistoryEntry(const HistoryEntry& other) : idx{other.idx}, ready{other.ready.load(std::memory_order_acquire)} {}

  bool operator==(const HistoryEntry& other) const { 
    return idx == other.idx && ready.load(std::memory_order_acquire) == other.ready.load(std::memory_order_acquire); 
  };

  HistoryEntry& operator=(const HistoryEntry &other) {
    idx = other.idx;
    ready.store(other.ready.load(std::memory_order_acquire), std::memory_order_release);
    return *this;
  }

  template <class Archive>
  void serialize(Archive& ar) {
    ar(idx, ready);
  }

  size_t idx;
  std::atomic<bool> ready;
};

template<class T>
class StrategyStorage {
public:
  StrategyStorage(const ActionProfile& action_profile, int n_clusters = 200) : 
               _action_profile{action_profile}, _n_clusters{n_clusters} {};

  StrategyStorage(int n_players = 2, int n_clusters = 200) : StrategyStorage{BlueprintActionProfile{n_players}, n_clusters} {}

  StrategyStorage(const StrategyStorage& other) 
      : _data(other._data), 
        _history_map(other._history_map), 
        _action_profile(other._action_profile), 
        _n_clusters(other._n_clusters) {
  }

  StrategyStorage(StrategyStorage&& other) noexcept 
      : _data(std::move(other._data)), 
        _history_map(std::move(other._history_map)), 
        _action_profile(std::move(other._action_profile)), 
        _n_clusters(other._n_clusters) {
  }

  inline const tbb::concurrent_vector<std::atomic<T>>& data() const { return _data; }
  inline tbb::concurrent_vector<std::atomic<T>>& data() { return _data; }
  inline const tbb::concurrent_unordered_map<ActionHistory, HistoryEntry>& history_map() const { return _history_map; }
  inline tbb::concurrent_unordered_map<ActionHistory, HistoryEntry>& history_map() { return _history_map; }
  inline const ActionProfile& action_profile() const { return _action_profile; }
  inline int n_clusters() const { return _n_clusters; }

  std::atomic<T>& operator[](size_t idx) { 
    if(idx >= _data.size()) throw std::runtime_error("Storage access out of bounds.");
    return _data[idx]; 
  }
  const std::atomic<T>& operator[](size_t idx) const { 
    if(idx >= _data.size()) throw std::runtime_error("Constant storage access out of bounds.");
    return _data[idx]; 
  }

  size_t index(const PokerState& state, int cluster, int action = 0) {
    auto actions = valid_actions(state, _action_profile);
    size_t n_actions = actions.size();
    auto history = state.get_action_history();
  
    // Fast path: no lock if already allocated and marked ready.
    if (auto it = _history_map.find(history); it != _history_map.end() && it->second.ready.load(std::memory_order_acquire)) {
      return it->second.idx + cluster * n_actions + action;
    }
  
    // Double-lock pattern: acquire the lock and re-check.
    std::unique_lock<std::mutex> lock(_grow_mutex);
    auto it = _history_map.find(history);
    if (it == _history_map.end()) {
      // First thread to handle this history.
      size_t history_idx = _data.size();
      HistoryEntry new_entry(history_idx, false);
  
      auto result = _history_map.emplace(history, new_entry);
      auto inserted_it = result.first;
  
      // Fully allocate memory for this history.
      _data.grow_by(_n_clusters * n_actions);
  
      // Mark as ready and notify waiting threads.
      inserted_it->second.ready.store(true, std::memory_order_release);
      _grow_cv.notify_all();
      return history_idx + cluster * n_actions + action;
    } else {
      // Another thread is handling allocation; wait until it's done.
      _grow_cv.wait(lock, [&]() {
        return it->second.ready.load(std::memory_order_acquire);
      });
      return it->second.idx + cluster * n_actions + action;
    }
  }

  size_t index(const PokerState& state, int cluster, int action = 0) const {
    size_t n_actions = valid_actions(state, _action_profile).size();
    auto it = _history_map.find(state.get_action_history());
    if(it != _history_map.end()) return it->second.idx + cluster * n_actions + action;
    throw std::runtime_error("StrategyStorage --- Indexed with unknown action history:\n" + state.get_action_history().to_string());
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
  tbb::concurrent_unordered_map<ActionHistory, HistoryEntry> _history_map;
  ActionProfile _action_profile;
  int _n_clusters;
  std::mutex _grow_mutex;
  std::condition_variable _grow_cv;
};

}