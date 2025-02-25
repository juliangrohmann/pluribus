#include <vector>
#include <pluribus/rng.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/infoset.hpp>
#include <pluribus/mccfr.hpp>
#include <pluribus/agent.hpp>

namespace pluribus {

Action RandomAgent::act(const PokerState& state, const Board& board, const Hand& hand, int n_players, int n_chips, int ante) {
  std::vector<Action> actions = valid_actions(state);
  assert(actions.size() > 0 && "No valid actions available.");
  std::uniform_int_distribution<int> dist(0, actions.size() - 1);
  return actions[dist(GlobalRNG::instance())];
}


BlueprintAgent::BlueprintAgent(BlueprintTrainer& trainer) {
  populate(PokerState{trainer.get_n_players(), trainer.get_n_chips(), trainer.get_ante()}, trainer);
}

Action BlueprintAgent::act(const PokerState& state, const Board& board, const Hand& hand, int n_players, int n_chips, int ante) {
  InformationSet info_set{state.get_action_history(), board, hand, state.get_round(), n_players, n_chips, ante};
  return _strategy.at(info_set);
}

void BlueprintAgent::populate(const PokerState& state, BlueprintTrainer& trainer) {
  int n_clusters = state.get_round() == 0 ? 169 : 200;
  for(uint16_t c = 0; c < n_clusters; ++c) {
    InformationSet info_set{state.get_action_history(), c, trainer.get_n_players(), trainer.get_n_chips(), trainer.get_ante()};
    StrategyMap& strategy_map = state.get_round() == 0 ? trainer.get_phi() : trainer.get_regrets();
    auto action_it = strategy_map.find(info_set);
    FrequencyMap freq;
    if(action_it != strategy_map.end()) {
      freq = calculate_strategy(action_it->second, state);
    }
    else {
      ActionMap action_map;
      freq = calculate_strategy(action_map, state);
    }
    _strategy[info_set] = sample_action(freq);
  }
  for(Action a : valid_actions(state)) {
    populate(state.apply(a), trainer);
  }
}

}