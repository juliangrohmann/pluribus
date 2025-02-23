#pragma once

#include <cereal/cereal.hpp>
#include <tbb/concurrent_unordered_map.h>
#include <pluribus/poker.hpp>
#include <pluribus/infoset.hpp>

namespace pluribus {

struct StrategyState {
  StrategyState() : regret{0}, frequency{0.0}, phi{0.0} {};
  int regret;
  float frequency;
  float phi;
  
  bool operator==(const StrategyState& other) const;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(regret, frequency, phi);
  }
};

using ActionMap = tbb::concurrent_unordered_map<Action, StrategyState>;
using StrategyMap = tbb::concurrent_unordered_map<InformationSet, ActionMap>;

const int it_per_sec = 24'000;
const int it_per_min = it_per_sec * 60;

class BlueprintTrainer {
public:
  BlueprintTrainer(int n_players = 6, int n_chips = 10'000, int ante = 0, long strategy_interval = 10'000, long preflop_threshold = 800 * it_per_min, 
                   long snapshot_interval = 200 * it_per_min, long prune_thresh = 200 * it_per_min, int prune_cutoff = -300'000'000, int regret_floor = -310'000'000,
                   long lcfr_thresh = 400 * it_per_min, long discount_interval = 10 * it_per_min, long log_interval = it_per_min);
  void set_strategy(const StrategyMap& strategy) { _strategy = strategy; }
  void mccfr_p(long T);
  long count_infosets();
  inline const StrategyMap& get_strategy() { return _strategy; }

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_strategy, _t, _strategy_interval, _preflop_threshold, _snapshot_interval, _prune_thresh, _lcfr_thresh, _discount_interval, _log_interval,
       _prune_cutoff, _regret_floor, _n_players, _n_chips, _ante);
  }

private:
  int traverse_mccfr_p(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands);
  int traverse_mccfr(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands);
  void update_strategy(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands);
  void calculate_strategy(const PokerState& state, const InformationSet& info_set);
  int utility(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands) const;
  int showdown_payoff(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands) const;
  Action sample(const InformationSet& info_set) const;
  void log_metrics(int t);

#ifdef UNIT_TEST
  friend int call_traverse_mccfr(BlueprintTrainer& trainer, const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands);
  friend void call_update_strategy(BlueprintTrainer& trainer, const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands);
#endif

  StrategyMap _strategy;
  omp::HandEvaluator _eval;
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

Action sample_action(const StrategyMap& strategy, const InformationSet& info_set);

}

namespace cereal {

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