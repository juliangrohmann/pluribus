#include <vector>
#include <pluribus/rng.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/infoset.hpp>
#include <pluribus/mccfr.hpp>
#include <pluribus/agent.hpp>

namespace pluribus {

Action RandomAgent::act(const PokerState& state, const Board& board, const Hand& hand) {
  std::vector<Action> actions = valid_actions(state);
  assert(actions.size() > 0 && "No valid actions available.");
  std::uniform_int_distribution<int> dist(0, actions.size() - 1);
  return actions[dist(GlobalRNG::instance())];
}

Action BlueprintAgent::act(const PokerState& state, const Board& board, const Hand& hand) {
  InformationSet info_set{state.get_action_history(), board, hand, state.get_round()};
  return sample_action(_strategy, info_set);
}

}