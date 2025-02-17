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

class BlueprintTrainer {
public:
  BlueprintTrainer(int n_players, int n_chips, int ante, int strategy_interval = 10'000, int prune_thresh = 12'000'000, 
                   int prune_cutoff = -300'000'000, int regret_floor = -310'000'000, int lcfr_thresh = 24'000'000, 
                   int discount_interval = 600'000, int log_interval = 10'000);
  void mccfr_p(int T);
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

  std::unordered_map<InformationSet, std::unordered_map<Action, StrategyState>> _strategy;
  omp::HandEvaluator _eval;
  long _cum_regret = 0;
  int _n_players;
  int _n_chips;
  int _ante;
  int _strategy_interval;
  int _prune_thresh;
  int _prune_cutoff;
  int _regret_floor;
  int _lcfr_thresh;
  int _discount_interval;
  int _log_interval;
};

}
