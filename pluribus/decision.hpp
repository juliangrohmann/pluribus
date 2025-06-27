#pragma once

#include <memory>
#include <pluribus/poker.hpp>
#include <pluribus/cluster.hpp>
#include <pluribus/storage.hpp>
#include <pluribus/actions.hpp>

namespace pluribus {

class DecisionAlgorithm {
public:
  virtual float frequency(Action a, const PokerState& state, const Board& board, const Hand& hand) const = 0;
};

template<class T>
class StrategyDecision : public DecisionAlgorithm {
public:
  StrategyDecision(const StrategyStorage<T>& strategy, const ActionProfile& profile) : _strategy{strategy}, _profile{profile} {}

  float frequency(Action a, const PokerState& state, const Board& board, const Hand& hand) const override {
    auto actions = valid_actions(state, _profile);
    int cluster = FlatClusterMap::get_instance()->cluster(state.get_round(), board, hand);
    int base_idx = _strategy.index(state, cluster);
    auto freq = calculate_strategy(_strategy, base_idx, actions.size());
    int a_idx = std::distance(actions.begin(), std::find(actions.begin(), actions.end(), a));
    return freq[a_idx];
  }

private:
  const StrategyStorage<T>& _strategy;
  ActionProfile _profile;
};

}