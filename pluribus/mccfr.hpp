#pragma once

#include <unordered_map>
#include <cereal/cereal.hpp>
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

const int it_per_sec = 3000;
const int it_per_min = it_per_sec * 60;

class BlueprintTrainer {
public:
  BlueprintTrainer(int n_players, int n_chips, int ante, int strategy_interval = 10'000, int preflop_threshold = 800 * it_per_min, 
                   int snapshot_interval = 200 * it_per_min, int prune_thresh = 200 * it_per_min, int prune_cutoff = -300'000'000, int regret_floor = -310'000'000,
                   int lcfr_thresh = 400 * it_per_min, int discount_interval = 10 * it_per_min, int log_interval = it_per_min);
  void mccfr_p(long T);
  void save_strategy(std::string fn) const;
  void load_strategy(std::string fn);
  long count_infosets();
  inline const std::unordered_map<InformationSet, std::unordered_map<Action, StrategyState>>& get_strategy() { return _strategy; }
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

  std::unordered_map<InformationSet, std::unordered_map<Action, StrategyState>> _strategy;
  omp::HandEvaluator _eval;
  long _cum_regret = 0;
  int _n_players;
  int _n_chips;
  int _ante;
  int _strategy_interval;
  int _preflop_threshold;
  int _snapshot_interval;
  int _prune_thresh;
  int _prune_cutoff;
  int _regret_floor;
  int _lcfr_thresh;
  int _discount_interval;
  int _log_interval;
};

}
