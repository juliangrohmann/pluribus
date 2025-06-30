#pragma once

#include <memory>
#include <pluribus/poker.hpp>
#include <pluribus/cluster.hpp>
#include <pluribus/storage.hpp>
#include <pluribus/tree_storage.hpp>
#include <pluribus/blueprint.hpp>
#include <pluribus/actions.hpp>
#include <pluribus/calc.hpp>

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
    auto freq = calculate_strategy(&_strategy[base_idx], actions.size());
    auto a_it = std::find(actions.begin(), actions.end(), a);
    if(a_it == actions.end()) Logger::error("Action " + a.to_string() + " is not in the action profile or not valid.");
    int a_idx = std::distance(actions.begin(), a_it);
    return freq[a_idx];
  }

private:
  const StrategyStorage<T>& _strategy;
  ActionProfile _profile;
};

template<class T>
class TreeDecision : public DecisionAlgorithm {
public:
  TreeDecision(const TreeStorageNode<T>* root, const PokerState& init_state) : _root{root}, _init_state{init_state} {}

  float frequency(Action a, const PokerState& state, const Board& board, const Hand& hand) const override {
    if(!state.get_action_history().is_consistent(_init_state.get_action_history())) {
    Logger::error("Cannot compute TreeSolver frequency for inconsistent histories:\nInitial state: " 
        + _init_state.get_action_history().to_string() + "\nGiven state: " + state.get_action_history().to_string());
    }
    const TreeStorageNode<T>* node = _root;
    for(int i = _init_state.get_action_history().size(); i < state.get_action_history().size(); ++i) {
      node = node->apply(state.get_action_history().get(i));
    }
    int cluster = FlatClusterMap::get_instance()->cluster(state.get_round(), board, hand);
    auto freq = calculate_strategy(node->get(cluster), node->get_actions().size());
    return freq[index_of(a, node->get_actions())];
  }

private:
  const PokerState _init_state;
  const TreeStorageNode<T>* _root;
};

template <class BlueprintT>
class ActionProvider {
public:
  virtual Action next_action(CachedIndexer& indexer, const PokerState& state, const std::vector<Hand>& hands, const Board& board, const BlueprintT* bp) const = 0;
};

class LosslessActionProvider : public ActionProvider<LosslessBlueprint> {
public:
  Action next_action(CachedIndexer& indexer, const PokerState& state, const std::vector<Hand>& hands, const Board& board, const LosslessBlueprint* bp) const override {
    auto actions = valid_actions(state, bp->get_config().action_profile);
    hand_index_t hand_idx = indexer.index(board, hands[state.get_active()], static_cast<int>(state.get_round()));
    int cluster = FlatClusterMap::get_instance()->cluster(state.get_round(), hand_idx);
    size_t base_idx = bp->get_strategy().index(state, cluster);
    auto freq = calculate_strategy(&bp->get_strategy()[base_idx], actions.size());
    return actions[sample_action_idx(freq)];
  }
};

class SampledActionProvider : public ActionProvider<SampledBlueprint> {
public:
  Action next_action(CachedIndexer& indexer, const PokerState& state, const std::vector<Hand>& hands, const Board& board, const SampledBlueprint* bp) const override {
    hand_index_t hand_idx = indexer.index(board, hands[state.get_active()], static_cast<int>(state.get_round()));
    int cluster = FlatClusterMap::get_instance()->cluster(state.get_round(), hand_idx);
    size_t base_idx = bp->get_strategy().index(state, cluster);
    uint8_t bias_offset = bp->bias_offset(state.get_biases()[state.get_active()]);
    return bp->get_strategy()[base_idx + bias_offset];
  }
};

}