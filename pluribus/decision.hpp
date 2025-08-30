#pragma once

#include <pluribus/actions.hpp>
#include <pluribus/blueprint.hpp>
#include <pluribus/calc.hpp>
#include <pluribus/cluster.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/tree_storage.hpp>

namespace pluribus {

class DecisionAlgorithm {
public:
  virtual ~DecisionAlgorithm() = default;

  virtual float frequency(Action a, const PokerState& state, const Board& board, const Hand& hand) const = 0;
};

template<class T>
class TreeDecision : public DecisionAlgorithm {
public:
  TreeDecision(const TreeStorageNode<T>* root, const PokerState& init_state, const bool real_time)
      : _init_state{init_state}, _root{root}, _real_time{real_time} {}

  float frequency(Action a, const PokerState& state, const Board& board, const Hand& hand) const override {
    if(!state.get_action_history().is_consistent(_init_state.get_action_history())) {
      Logger::error("Cannot compute TreeSolver frequency for inconsistent histories:\nInitial state: "
        + _init_state.get_action_history().to_string() + "\nGiven state: " + state.get_action_history().to_string());
    }
    const TreeStorageNode<T>* node = _root;
    for(int i = _init_state.get_action_history().size(); i < state.get_action_history().size(); ++i) {
      node = node->apply(state.get_action_history().get(i));
    }
    int cluster = _real_time ?
        (state.get_round() == _init_state.get_round() ?
            HoleCardIndexer::get_instance()->index(hand) :
            RealTimeClusterMap::get_instance()->cluster(state.get_round(), board, hand)) :
        BlueprintClusterMap::get_instance()->cluster(state.get_round(), board, hand);
    auto freq = calculate_strategy(node->get(cluster), node->get_value_actions().size());
    if(std::ranges::find(node->get_value_actions(), a) == node->get_value_actions().end()) {
      std::cout << "Failed to find action: " << a.to_string();
      std::cout << "Value actions:\n";
      for(Action va : node->get_value_actions()) {
        std::cout << va.to_string() << "\n";
      }
      std::cout << "Init state: " << _init_state.get_action_history().to_string() << "\n";
      std::cout << "Curr state: " << state.get_action_history().to_string() << "\n";
    }
    return freq[index_of(a, node->get_value_actions())];
  }

private:
  const PokerState _init_state;
  const TreeStorageNode<T>* _root;
  const bool _real_time;
};

template <class BlueprintT>
class ActionProvider {
public:
  virtual ~ActionProvider() = default;

  virtual Action next_action(CachedIndexer& indexer, const PokerState& state, const std::vector<Hand>& hands, const Board& board, const BlueprintT* bp) const = 0;
};

class LosslessActionProvider : public ActionProvider<LosslessBlueprint> {
public:
  Action next_action(CachedIndexer& indexer, const PokerState& state, const std::vector<Hand>& hands, const Board& board,
    const LosslessBlueprint* bp) const override;
};

class SampledActionProvider : public ActionProvider<SampledBlueprint> {
public:
  Action next_action(CachedIndexer& indexer, const PokerState& state, const std::vector<Hand>& hands, const Board& board,
    const SampledBlueprint* bp) const override;
};

}