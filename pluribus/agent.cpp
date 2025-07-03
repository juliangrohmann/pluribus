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
  const std::vector<Action> actions = valid_actions(state, _action_profile);
  assert(actions.size() > 0 && "No valid actions available.");
  std::uniform_int_distribution dist(0, static_cast<int>(actions.size()) - 1);
  return actions[dist(GlobalRNG::instance())];
}

Action BlueprintAgent::act(const PokerState& state, const Board& board, const Hand& hand, const PokerConfig& config) {
  const auto actions = valid_actions(state, _trainer_p->get_config().action_profile);
  const int cluster = FlatClusterMap::get_instance()->cluster(state.get_round(), board, hand);
  const size_t base_idx = _trainer_p->get_strategy().index(state, cluster);
  const auto freq = calculate_strategy(&_trainer_p->get_strategy()[base_idx], static_cast<int>(actions.size()));
  return actions[sample_action_idx(freq)];
}

template <class T>
std::vector<T> shift(const std::vector<T>& data, const int n) {
  std::vector<T> shifted;
  for(int i = n; i < n + static_cast<int>(data.size()); ++i) {
    if(i < 0) shifted.push_back(data[i + data.size()]);
    else if(i >= data.size()) shifted.push_back(data[i - data.size()]);
    else shifted.push_back(data[i]);
  }
  return shifted;
}

void evaluate_agents(const std::vector<Agent*>& agents, const PokerConfig& config, const long n_iter) {
  const long iter_per_pos = n_iter / static_cast<long>(agents.size());
  std::vector winrates(6, 0.0);
  for(int i = 0; i < agents.size(); ++i) {
    auto shifted_agents = shift(agents, i);
    auto shifted_results = simulate(shifted_agents, config, iter_per_pos);
    auto results = shift(shifted_results, -i);
    for(int j = 0; j < agents.size(); ++j) {
      winrates[j] += static_cast<double>(results[j]) / static_cast<double>(iter_per_pos);
    }
  }
  for(int i = 0; i < agents.size(); ++i) {
    std::cout << "Player " << i << ": " << std::setprecision(2) << std::fixed << winrates[i] / static_cast<double>(agents.size()) << " bb/100\n";
  }
}

void evaluate_strategies(const std::vector<MappedBlueprintSolver*>& strategies, const long n_iter) {
  std::vector<BlueprintAgent> agents;
  for(const auto& p : strategies) agents.emplace_back(p);
  std::vector<Agent*> agent_ps;
  for(auto& agent : agents) agent_ps.push_back(&agent);
  evaluate_agents(agent_ps, strategies[0]->get_config().poker, n_iter);
}

void evaluate_vs_random(const MappedBlueprintSolver* hero, const long n_iter) {
  BlueprintAgent bp_agent{hero};
  std::vector<RandomAgent> rng_agents;
  for(int i = 0; i < hero->get_config().poker.n_players - 1; ++i) rng_agents.emplace_back(hero->get_config().action_profile);
  std::vector<Agent*> agents_p{&bp_agent};
  for(auto& agent : rng_agents) agents_p.push_back(&agent);
  evaluate_agents(agents_p, hero->get_config().poker, n_iter);
}

}