#pragma once

#include <array>
#include <vector>
#include <atomic>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cereal/cereal.hpp>
#include <tbb/concurrent_vector.h>
#include <tbb/concurrent_unordered_map.h>
#include <pluribus/poker.hpp>
#include <pluribus/infoset.hpp>

namespace pluribus {

using PreflopMap = tbb::concurrent_unordered_map<InformationSet, tbb::concurrent_vector<float>>;

class RegretStorage {
public:
  RegretStorage(int n_players = 2, int n_chips = 10'000, int ante = 0, int n_clusters = 200, int n_actions = 5);
  ~RegretStorage();
  std::atomic<int>* operator[](const InformationSet& info_set);
  const std::atomic<int>* operator[](const InformationSet& info_set) const;
  bool operator==(const RegretStorage& other) const;
  inline std::atomic<int>* data() { return _data; }
  inline const std::atomic<int>* data() const { return _data; }
  inline size_t size() { return _size; }
  inline int get_n_clusters() const { return _n_clusters; }

  void log(int n) {
    for(int i = 0; i < n; ++i) {
      std::cout << i << ": " << *(_data + i) << "\n";
    }
  }

  template <class Archive>
  void serialize(Archive& ar) {
    if(Archive::is_loading::value) {
      unmap_memory();
    }
    ar(_size, _n_clusters, _n_actions);
    if(Archive::is_loading::value) {
      map_memory();
    }
    ar(cereal::binary_data(_data, _size * sizeof(std::atomic<int>)));
  }

private:
  size_t info_offset(const InformationSet& info_set) const;
  void map_memory();
  void unmap_memory();
  
  std::atomic<int>* _data;
  size_t _size;
  std::string _fn;
  int _n_clusters;
  int _n_actions;
  int _fd;
};

std::vector<float> calculate_strategy(const std::atomic<int>* regret_p, int n_actions);
int sample_action_idx(const std::vector<float>& freq);
void lcfr_discount(RegretStorage& strategy, double d);
void lcfr_discount(PreflopMap& strategy, double d);

class BlueprintTrainer {
public:
  BlueprintTrainer(int n_players = 2, int n_chips = 10'000, int ante = 0, long strategy_interval = 10'000, long preflop_threshold_m = 800, 
                   long snapshot_interval_m = 200, long prune_thresh_m = 200, int prune_cutoff = -300'000'000, 
                   int regret_floor = -310'000'000, long lcfr_thresh_m = 400, long discount_interval_m = 10, 
                   long log_interval_m = 1, long profiling_thresh = 1'000'000);
  void mccfr_p(long T);
  void log_state() const;
  bool operator==(const BlueprintTrainer& other);
  inline const RegretStorage& get_regrets() const { return _regrets; }
  inline RegretStorage& get_regrets() { return _regrets; }
  inline const PreflopMap& get_phi() const { return _phi; }
  inline int get_n_players() const { return _n_players; }
  inline int get_n_chips() const { return _n_chips; }
  inline int get_ante() const { return _ante; }
  
  template <class Archive>
  void serialize(Archive& ar) {
    ar(_regrets, _phi, _t, _strategy_interval, _preflop_threshold_m, _snapshot_interval_m, _prune_thresh_m, _lcfr_thresh_m, _discount_interval_m,
       _log_interval_m, _prune_cutoff, _regret_floor, _n_players, _n_chips, _ante);
  }

private:
  int traverse_mccfr_p(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, const omp::HandEvaluator& eval);
  int traverse_mccfr(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, const omp::HandEvaluator& eval);
  void update_strategy(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands);
  int utility(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, const omp::HandEvaluator& eval) const;
  int showdown_payoff(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, const omp::HandEvaluator& eval) const;
  void log_metrics(long t) const;

#ifdef UNIT_TEST
  friend int call_traverse_mccfr(BlueprintTrainer& trainer, const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, 
                                 const omp::HandEvaluator& eval);
  friend void call_update_strategy(BlueprintTrainer& trainer, const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands);
#endif

  RegretStorage _regrets;
  PreflopMap _phi;
  long _t;
  long _strategy_interval;
  long _preflop_threshold_m;
  long _snapshot_interval_m;
  long _prune_thresh_m;
  long _lcfr_thresh_m;
  long _discount_interval_m;
  long _log_interval_m;
  long _it_per_min;
  long _profiling_thresh;
  int _prune_cutoff;
  int _regret_floor;
  int _n_players;
  int _n_chips;
  int _ante;
};

}

namespace cereal {

template<class Archive>
void serialize(Archive& archive, std::atomic<int>& atomic_int) {
    int value = atomic_int.load(std::memory_order_relaxed);
    archive(value);
    if(Archive::is_loading::value) {
      atomic_int.store(value, std::memory_order_relaxed);
    }
}

template<class Archive, class T>
void save(Archive& ar, const tbb::concurrent_vector<T>& vec) {
  size_t size = vec.size();
  ar(size);
  for(size_t i = 0; i < size; ++i) {
    ar(vec[i]);
  }
}

template<class Archive, class T>
void load(Archive& ar, tbb::concurrent_vector<T>& vec) {
  size_t size;
  ar(size);
  vec.clear();
  auto it = vec.grow_by(size);
  for(size_t i = 0; i < size; ++i) {
    ar(*it);
    ++it;
  }
}

template<class Archive, class Key, class T>
void save(Archive& ar, const tbb::concurrent_unordered_map<Key, T>& map) {
  size_t size = map.size();
  ar(size);
  for(const auto& pair : map) {
    ar(pair.first, pair.second);
  }
}

template<class Archive, class Key, class T>
void load(Archive& ar, tbb::concurrent_unordered_map<Key, T>& map) {
  size_t size;
  ar(size);
  map.clear();
  for(size_t i = 0; i < size; ++i) {
    Key key;
    T value;
    ar(key, value);
    map[key] = value;
  }
}

}