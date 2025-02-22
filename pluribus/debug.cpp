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
                         const ActionHistory& history, Action action, const Board& board, int round) {
  std::ostringstream oss;
  for(uint8_t i = 0; i < 52; ++i) {
    for(uint8_t j = i + 1; j < 52; ++j) {
      Hand hand{j, i};
      InformationSet info_set{history, board, hand, round};
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

void evaluate_strategy(const StrategyMap& hero, const StrategyMap& villain) {
  long n_iter = 1'000'000;
  std::vector<BlueprintAgent> agents{BlueprintAgent{hero}, BlueprintAgent{villain}};
  std::vector<Agent*> agents_p;
  for(auto& agent : agents) agents_p.push_back(&agent);
  auto results = simulate(agents_p, 10'000, 0, n_iter);
  std::cout << "Hero: " << (results[0][results[0].size() - 1] / static_cast<double>(n_iter)) * 100 << "bb/100\n";
  std::cout << "Villain: " << (results[1][results[1].size() - 1] / static_cast<double>(n_iter)) * 100 << "bb/100\n";
}

}