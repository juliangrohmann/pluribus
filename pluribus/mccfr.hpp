#pragma once

#include <array>
#include <vector>
#include <atomic>
#include <cereal/cereal.hpp>
#include <tbb/concurrent_vector.h>
#include <tbb/concurrent_unordered_map.h>
#include <pluribus/poker.hpp>
#include <pluribus/infoset.hpp>

namespace pluribus {

using PreflopMap = tbb::concurrent_unordered_map<InformationSet, tbb::concurrent_vector<float>>;

struct atom_int {
  atom_int() noexcept : value(0) {}
  atom_int(int val) noexcept : value(val) {}
  atom_int(const atom_int& other) noexcept : value(other.value.load()) {}
  atom_int(atom_int&& other) noexcept : value(other.value.load()) {}

  atom_int& operator=(atom_int&& other) noexcept {
    value.store(other.value.load());
    return *this;
  }
  atom_int& operator=(const atom_int&) = delete;
  bool operator==(const atom_int& other) const { return value.load() == other.load(); }

  std::atomic<int> value;

  template <class Archive>
  void serialize(Archive& ar) {
    int tmp = value.load(); // Get the current value for output
    ar(tmp);                // Serialize/deserialize the plain int
    if constexpr (std::is_same_v<Archive, cereal::BinaryInputArchive>) {
      value.store(tmp);     // Only store on input
    }
  }

  int load() const noexcept { return value.load(); }
  void store(int val) noexcept { value.store(val); }
};

class RegretStorage {
public:
  RegretStorage(int n_players, int n_chips, int ante, int n_clusters, int n_actions);
  std::atomic<int>& operator[](const InformationSet& info_set);
  const std::atomic<int>& operator[](const InformationSet& info_set) const;
  bool operator==(const RegretStorage& other) const;
  inline std::vector<atom_int>& data() { return _data; }
  inline const std::vector<atom_int>& data() const { return _data; }
  inline size_t size() { return _data.size(); }
  inline int get_n_clusters() const { return _n_clusters; }

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_data, _n_clusters, _n_actions);
  }
private:
  size_t idx(const InformationSet& info_set) const;
  
  std::vector<atom_int> _data;
  int _n_clusters;
  int _n_actions;
};

std::vector<float> calculate_strategy(std::atomic<int>* regret_p, int n_actions);
int sample_action_idx(const std::vector<float>& freq);
void lcfr_discount(RegretStorage& strategy, double d);
void lcfr_discount(PreflopMap& strategy, double d);

const int it_per_sec = 24'000;
const int it_per_min = it_per_sec * 60;

class BlueprintTrainer {
public:
  BlueprintTrainer(int n_players = 6, int n_chips = 10'000, int ante = 0, long strategy_interval = 10'000, long preflop_threshold = 800 * it_per_min, 
                   long snapshot_interval = 200 * it_per_min, long prune_thresh = 200 * it_per_min, int prune_cutoff = -300'000'000, 
                   int regret_floor = -310'000'000, long lcfr_thresh = 400 * it_per_min, long discount_interval = 10 * it_per_min, 
                   long log_interval = it_per_min);
  void mccfr_p(long T);
  void log_state() const;
  inline const RegretStorage& get_regrets() const { return _regrets; }
  inline RegretStorage& get_regrets() { return _regrets; }
  inline const PreflopMap& get_phi() const { return _phi; }
  inline int get_n_players() const { return _n_players; }
  inline int get_n_chips() const { return _n_chips; }
  inline int get_ante() const { return _ante; }

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_regrets, _phi, _t, _strategy_interval, _preflop_threshold, _snapshot_interval, _prune_thresh, _lcfr_thresh, _discount_interval,
       _log_interval,_prune_cutoff, _regret_floor, _n_players, _n_chips, _ante);
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
  long _preflop_threshold;
  long _snapshot_interval;
  long _prune_thresh;
  long _lcfr_thresh;
  long _discount_interval;
  long _log_interval;
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
    atomic_int.store(value, std::memory_order_relaxed);
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