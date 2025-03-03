#include <vector>
#include <pluribus/rng.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/infoset.hpp>
#include <pluribus/mccfr.hpp>
#include <pluribus/agent.hpp>

namespace pluribus {

Action RandomAgent::act(const PokerState& state, const Board& board, const Hand& hand, int n_players, int n_chips, int ante) {
  std::vector<Action> actions = valid_actions(state, _action_profile);
  assert(actions.size() > 0 && "No valid actions available.");
  std::uniform_int_distribution<int> dist(0, actions.size() - 1);
  return actions[dist(GlobalRNG::instance())];
}

Action BlueprintAgent::act(const PokerState& state, const Board& board, const Hand& hand, int n_players, int n_chips, int ante) {
  InformationSet info_set{state.get_action_history(), board, hand, state.get_round(), n_players, n_chips, ante};
  auto actions = valid_actions(state, _trainer_p->get_action_profile());
  auto freq = calculate_strategy(_trainer_p->get_regrets()[info_set], actions.size());
  return actions[sample_action_idx(freq)];
}

SampledBlueprintAgent::SampledBlueprintAgent(const BlueprintTrainer& trainer) : 
    _strategy{trainer.get_n_players(), trainer.get_n_chips(), trainer.get_ante(), trainer.get_regrets().get_n_clusters()}, _action_profile{trainer.get_action_profile()} {
  std::cout << "Populating sampled blueprint...\n";
  populate(PokerState{trainer.get_n_players(), trainer.get_n_chips(), trainer.get_ante()}, trainer);
}

Action SampledBlueprintAgent::act(const PokerState& state, const Board& board, const Hand& hand, int n_players, int n_chips, int ante) {
  InformationSet info_set{state.get_action_history(), board, hand, state.get_round(), n_players, n_chips, ante};
  return _strategy[info_set];
}

std::vector<float> calculate_strategy(const tbb::concurrent_vector<float>& phi_map, int n_actions) {
  float sum = 0;
  for(int a_idx = 0; a_idx < n_actions; ++a_idx) {
    sum += std::max(phi_map[a_idx], 0.0f);
  }

  std::vector<float> freq;
  freq.reserve(n_actions);
  if(sum > 0.0f) {
    for(int a_idx = 0; a_idx < n_actions; ++a_idx) {
      freq.push_back(std::max(phi_map[a_idx], 0.0f) / static_cast<double>(sum));
    }
  }
  else {
    for(int i = 0; i < n_actions; ++i) {
      freq.push_back(1.0 / n_actions);
    }
  }
  return freq;
}

void SampledBlueprintAgent::populate(const PokerState& state, const BlueprintTrainer& trainer) {
  if(state.is_terminal()) return;

  int n_clusters = state.get_round() == 0 ? 169 : 200;
  for(uint16_t c = 0; c < n_clusters; ++c) {
    InformationSet info_set{state.get_action_history(), c, trainer.get_n_players(), trainer.get_n_chips(), trainer.get_ante()};
    auto actions = valid_actions(state, _action_profile);
    std::vector<float> freq;
    if(state.get_round() == 0) {
      freq = calculate_strategy(trainer.get_phi().at(info_set), actions.size());
    }
    else {
      freq = calculate_strategy(trainer.get_regrets()[info_set], actions.size());
    }
    int a_idx = sample_action_idx(freq);
    _strategy[info_set] = actions[a_idx];
  }
  for(Action a : valid_actions(state, _action_profile)) {
    populate(state.apply(a), trainer);
  }
}

}