#include <vector>
#include <pluribus/rng.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/indexing.hpp>
#include <pluribus/mccfr.hpp>
#include <pluribus/cluster.hpp>
#include <pluribus/simulate.hpp>
#include <pluribus/agent.hpp>

namespace pluribus {

Action RandomAgent::act(const PokerState& state, const Board& board, const Hand& hand, const PokerConfig& config) {
  std::vector<Action> actions = valid_actions(state, _action_profile);
  assert(actions.size() > 0 && "No valid actions available.");
  std::uniform_int_distribution<int> dist(0, actions.size() - 1);
  return actions[dist(GlobalRNG::instance())];
}

Action BlueprintAgent::act(const PokerState& state, const Board& board, const Hand& hand, const PokerConfig& config) {
  auto actions = valid_actions(state, _trainer_p->get_config().action_profile);
  int cluster = FlatClusterMap::get_instance()->cluster(state.get_round(), board, hand);
  size_t base_idx = _trainer_p->get_strategy().index(state, cluster);
  auto freq = calculate_strategy(_trainer_p->get_strategy(), base_idx, actions.size());
  return actions[sample_action_idx(freq)];
}

// SampledBlueprintAgent::SampledBlueprintAgent(const BlueprintSolver& trainer) : 
//     _strategy{trainer.get_config().poker, trainer.get_strategy().n_clusters()}, _action_profile{trainer.get_config().action_profile} {
//   std::cout << "Populating sampled blueprint...\n";
//   populate(PokerState{trainer.get_config().poker}, trainer);
// }

// Action SampledBlueprintAgent::act(const PokerState& state, const Board& board, const Hand& hand, const PokerConfig& config) {
//   InformationSet info_set{state.get_action_history(), board, hand, state.get_round(), config};
//   return _strategy[info_set];
// }

// void SampledBlueprintAgent::populate(const PokerState& state, const BlueprintSolver& trainer) {
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
//       size_t base_idx = trainer.get_strategy().index(state, c);
//       freq = calculate_strategy(trainer.get_strategy(), base_idx, actions.size());
//     }
//     int a_idx = sample_action_idx(freq);
//     _strategy[info_set] = actions[a_idx];
//   }
//   for(Action a : valid_actions(state, _action_profile)) {
//     populate(state.apply(a), trainer);
//   }
// }

template <class T>
std::vector<T> shift(const std::vector<T>& data, int n) {
  std::vector<T> shifted;
  for(int i = n; i < n + static_cast<int>(data.size()); ++i) {
    if(i < 0) shifted.push_back(data[i + data.size()]);
    else if(i >= data.size()) shifted.push_back(data[i - data.size()]);
    else shifted.push_back(data[i]);
  }
  return shifted;
}

void evaluate_agents(const std::vector<Agent*>& agents, const PokerConfig& config, long n_iter) {
  long iter_per_pos = n_iter / agents.size();
  std::vector<double> winrates(6, 0.0);
  for(int i = 0; i < agents.size(); ++i) {
    auto shifted_agents = shift(agents, i);
    auto shifted_results = simulate(shifted_agents, config, iter_per_pos);
    auto results = shift(shifted_results, -i);
    for(int j = 0; j < agents.size(); ++j) {
      winrates[j] += results[j] / static_cast<double>(iter_per_pos);
    }
  }
  for(int i = 0; i < agents.size(); ++i) {
    std::cout << "Player " << i << ": " << std::setprecision(2) << std::fixed << winrates[i] / agents.size() << " bb/100\n";
  }
}

void evaluate_strategies(const std::vector<BlueprintSolver*>& trainer_ps, long n_iter) {
  std::vector<BlueprintAgent> agents;
  for(const auto& p : trainer_ps) agents.push_back(BlueprintAgent{p});
  std::vector<Agent*> agent_ps;
  for(auto& agent : agents) agent_ps.push_back(&agent);
  evaluate_agents(agent_ps, trainer_ps[0]->get_config().poker, n_iter);
}

void evaluate_vs_random(const BlueprintSolver* trainer_p, long n_iter) {
  BlueprintAgent bp_agent{trainer_p};
  std::vector<RandomAgent> rng_agents;
  for(int i = 0; i < trainer_p->get_config().poker.n_players - 1; ++i) rng_agents.push_back(RandomAgent(trainer_p->get_config().action_profile));
  std::vector<Agent*> agents_p{&bp_agent};
  for(auto& agent : rng_agents) agents_p.push_back(&agent);
  evaluate_agents(agents_p, trainer_p->get_config().poker, n_iter);
}

}