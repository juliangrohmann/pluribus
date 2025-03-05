#include <vector>
#include <pluribus/rng.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/infoset.hpp>
#include <pluribus/mccfr.hpp>
#include <pluribus/cluster.hpp>
#include <pluribus/agent.hpp>

namespace pluribus {

Action RandomAgent::act(const PokerState& state, const Board& board, const Hand& hand, const PokerConfig& config) {
  std::vector<Action> actions = valid_actions(state, _action_profile);
  assert(actions.size() > 0 && "No valid actions available.");
  std::uniform_int_distribution<int> dist(0, actions.size() - 1);
  return actions[dist(GlobalRNG::instance())];
}

Action BlueprintAgent::act(const PokerState& state, const Board& board, const Hand& hand, const PokerConfig& config) {
  InformationSet info_set{state.get_action_history(), board, hand, state.get_round(), config};
  auto actions = valid_actions(state, _trainer_p->get_config().action_profile);
  int cluster = FlatClusterMap::get_instance()->cluster(state.get_round(), board, hand);
  size_t base_idx = _trainer_p->get_regrets().index(state, cluster);
  auto freq = calculate_strategy(_trainer_p->get_regrets(), base_idx, actions.size());
  return actions[sample_action_idx(freq)];
}

// SampledBlueprintAgent::SampledBlueprintAgent(const BlueprintTrainer& trainer) : 
//     _strategy{trainer.get_config().poker, trainer.get_regrets().n_clusters()}, _action_profile{trainer.get_config().action_profile} {
//   std::cout << "Populating sampled blueprint...\n";
//   populate(PokerState{trainer.get_config().poker}, trainer);
// }

// Action SampledBlueprintAgent::act(const PokerState& state, const Board& board, const Hand& hand, const PokerConfig& config) {
//   InformationSet info_set{state.get_action_history(), board, hand, state.get_round(), config};
//   return _strategy[info_set];
// }

// void SampledBlueprintAgent::populate(const PokerState& state, const BlueprintTrainer& trainer) {
//   if(state.is_terminal()) return;

//   int n_clusters = state.get_round() == 0 ? 169 : 200;
//   for(uint16_t c = 0; c < n_clusters; ++c) {
//     InformationSet info_set{state.get_action_history(), c, trainer.get_config().poker};
//     auto actions = valid_actions(state, _action_profile);
//     std::vector<float> freq;
//     if(state.get_round() == 0) {
//       freq = calculate_strategy(trainer.get_phi().at(info_set), actions.size());
//     }
//     else {
//       size_t base_idx = trainer.get_regrets().index(state, c);
//       freq = calculate_strategy(trainer.get_regrets(), base_idx, actions.size());
//     }
//     int a_idx = sample_action_idx(freq);
//     _strategy[info_set] = actions[a_idx];
//   }
//   for(Action a : valid_actions(state, _action_profile)) {
//     populate(state.apply(a), trainer);
//   }
// }

}