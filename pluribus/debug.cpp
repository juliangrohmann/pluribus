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

std::string strategy_str(const std::unordered_map<InformationSet, std::unordered_map<Action, StrategyState>>& strategy, 
                         const ActionHistory& history, Action action, const Board& board, int round, int n_players, int n_chips, int ante) {
  std::ostringstream oss;
  for(uint8_t i = 0; i < 52; ++i) {
    for(uint8_t j = i + 1; j < 52; ++j) {
      Hand hand{j, i};
      InformationSet info_set{history, board, hand, round, n_players, n_chips, ante};
      auto action_map_it = strategy.find(info_set);
      if(action_map_it == strategy.end()) {
        std::cout << cards_to_str(hand.cards().data(), 2) << ": Info set missing.\n";
        continue;
      }
      auto strat_it = action_map_it->second.find(action);
      if(strat_it == action_map_it->second.end()) {
        std::cout << cards_to_str(hand.cards().data(), 2) << ": Action missing.\n";
        continue;
      }
      float freq = strat_it->second.frequency * 100;
      if(freq < 0.1) {
        std::cout << cards_to_str(hand.cards().data(), 2) << ": Frequency=" << freq << "\n";
        continue;
      }
      oss << std::fixed << std::setprecision(1) << "[" << freq << "]" << cards_to_str(hand.cards().data(), 2) << "[/" << freq << "],";
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

void evaluate_agents(const std::vector<Agent*>& agents, long n_iter) {
  long iter_per_pos = n_iter / agents.size();
  std::vector<double> winrates(6, 0.0);
  for(int i = 0; i < agents.size(); ++i) {
    auto shifted_agents = shift(agents, i);
    auto shifted_results = simulate(shifted_agents, 10'000, 0, iter_per_pos);
    auto results = shift(shifted_results, -i);
    for(int j = 0; j < agents.size(); ++j) {
      winrates[j] += results[j] / static_cast<double>(iter_per_pos);
    }
  }
  for(int i = 0; i < agents.size(); ++i) {
    std::cout << "Player " << i << ": " << std::setprecision(2) << std::fixed << winrates[i] / agents.size() << " bb/100\n";
  }
}

void evaluate_strategies(const std::vector<StrategyMap>& strategies, long n_iter) {
  std::vector<BlueprintAgent> agents;
  for(const auto& strat : strategies) agents.push_back(BlueprintAgent{strat});
  std::vector<Agent*> agents_p;
  for(auto& agent : agents) agents_p.push_back(&agent);
  evaluate_agents(agents_p, n_iter);
}

void evaluate_vs_random(const StrategyMap& hero, int n_players, long n_iter) {
  BlueprintAgent bp_agent{hero};
  std::vector<RandomAgent> rng_agents(n_players - 1);
  std::vector<Agent*> agents_p{&bp_agent};
  for(auto& agent : rng_agents) agents_p.push_back(&agent);
  evaluate_agents(agents_p, n_iter);
}

}