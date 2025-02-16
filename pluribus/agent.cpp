#include <vector>
#include <random>
#include <pluribus/poker.hpp>
#include <pluribus/agent.hpp>

namespace pluribus {

Action RandomAgent::act(const PokerState& state) {
  std::vector<Action> actions = valid_actions(state);
  assert(actions.size() > 0 && "No valid actions available.");
  std::uniform_int_distribution<int> dist(0, actions.size() - 1);
  return actions[dist(_rng)];
}

}