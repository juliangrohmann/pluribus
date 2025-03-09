#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cnpy.h>
#include <hand_isomorphism/hand_index.h>
#include <pluribus/util.hpp>
#include <pluribus/cluster.hpp>
#include <pluribus/simulate.hpp>
#include <pluribus/mccfr.hpp>
#include <pluribus/debug.hpp>

namespace pluribus {

std::string round_to_str(int round) {
  switch(round) {
    case 0: return "Preflop";
    case 1: return "Flop";
    case 2: return "Turn";
    case 3: return "River";
    case 4: return "Showdown";
    default: return "Unknown round: " + std::to_string(round);
  }
}

const std::vector<std::string> positions{"BTN", "CO", "HJ", "LJ", "UTG+2", "UTG+1", "UTG", "BB", "SB"};
std::string pos_to_str(int pos, int n_players) {
  return positions[n_players - pos];
}

std::string cluster_file(int round, int n_clusters) {
  return "clusters_r" + std::to_string(round) + "_c" + std::to_string(n_clusters) + ".npy";
}

void print_cluster(int cluster, int round, const hand_indexer_t& indexer, const std::vector<int>& clusters) {
  int card_sum = round + 4;
  int n_idx = hand_indexer_size(&indexer, round);
  uint8_t cards[7];
  for(int i = 0; i < n_idx; ++i) {
    if(clusters[i] != cluster) continue;
    hand_unindex(&indexer, round, i, cards);
    std::cout << cards_to_str(cards, card_sum) << "\n";
  }
}

void print_cluster(int cluster, int round, int n_clusters) {
  std::vector<int> clusters = cnpy::npy_load(cluster_file(round, n_clusters)).as_vec<int>();
  hand_indexer_t indexer;
  int card_sum = init_indexer(indexer, round);
  print_cluster(cluster, round, indexer, clusters);
}

void print_similar_boards(std::string board, int n_clusters) {
  int round = board.length() / 2 - 4;
  std::vector<int> clusters = cnpy::npy_load(cluster_file(round, n_clusters)).as_vec<int>();
  hand_indexer_t indexer;
  int card_sum = init_indexer(indexer, round);

  uint8_t cards[7];
  str_to_cards(board, cards);
  hand_index_t idx = hand_index_last(&indexer, cards);
  int cluster = clusters[idx];
  std::cout << board << " cluster: " << cluster << std::endl;
  print_cluster(cluster, round, indexer, clusters);
}

std::string strategy_str(const BlueprintTrainer& trainer, const PokerState& state, Action action, const Board& board) {
  std::ostringstream oss;
  for(uint8_t i = 0; i < 52; ++i) {
    for(uint8_t j = i + 1; j < 52; ++j) {
      Hand hand{j, i};
      auto actions = valid_actions(state, trainer.get_config().action_profile);
      int cluster = FlatClusterMap::get_instance()->cluster(state.get_round(), board, hand);
      int base_idx = trainer.get_regrets().index(state, cluster);
      auto freq = calculate_strategy(trainer.get_regrets(), base_idx, actions.size());
      int a_idx = std::distance(actions.begin(), std::find(actions.begin(), actions.end(), action));
      oss << std::fixed << std::setprecision(1) << "[" << freq[a_idx] << "]" << cards_to_str(hand.cards().data(), 2) << "[/" << freq[a_idx] << "],";
    }
  }
  return oss.str();
}

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

void evaluate_strategies(const std::vector<BlueprintTrainer*>& trainer_ps, long n_iter) {
  std::vector<BlueprintAgent> agents;
  for(const auto& p : trainer_ps) agents.push_back(BlueprintAgent{p});
  std::vector<Agent*> agent_ps;
  for(auto& agent : agents) agent_ps.push_back(&agent);
  evaluate_agents(agent_ps, trainer_ps[0]->get_config().poker, n_iter);
}

void evaluate_vs_random(const BlueprintTrainer* trainer_p, long n_iter) {
  BlueprintAgent bp_agent{trainer_p};
  std::vector<RandomAgent> rng_agents;
  for(int i = 0; i < trainer_p->get_config().poker.n_players - 1; ++i) rng_agents.push_back(RandomAgent(trainer_p->get_config().action_profile));
  std::vector<Agent*> agents_p{&bp_agent};
  for(auto& agent : rng_agents) agents_p.push_back(&agent);
  evaluate_agents(agents_p, trainer_p->get_config().poker, n_iter);
}

}