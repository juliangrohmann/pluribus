#include <cnpy.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <hand_isomorphism/hand_index.h>
#include <pluribus/cluster.hpp>
#include <pluribus/debug.hpp>
#include <pluribus/util.hpp>

namespace pluribus {

std::string round_to_str(const int round) {
  switch(round) {
    case 0: return "Preflop";
    case 1: return "Flop";
    case 2: return "Turn";
    case 3: return "River";
    case 4: return "Showdown";
    default: return "Unknown round: " + std::to_string(round);
  }
}

const std::vector<std::string> positions{"BTN", "CO", "HJ", "LJ", "UTG+2", "UTG+1", "UTG"};
std::string pos_to_str(const size_t idx, const size_t n_players, const bool straddle) {
  if(idx == 0) return n_players == 2 ? "BB" : "SB";
  if(idx == 1) return n_players == 2 ? "SB" : "BB";
  if(idx == 2 && straddle) return "STR";
  return positions[n_players - idx - 1];
}

std::string cluster_file(const int round, const int n_clusters) {
  return "clusters_r" + std::to_string(round) + "_c" + std::to_string(n_clusters) + ".npy";
}

void print_cluster(const int cluster, const int round, const hand_indexer_t& indexer, const std::vector<int>& clusters) {
  const int card_sum = round + 4;
  const int n_idx = hand_indexer_size(&indexer, round);
  uint8_t cards[7];
  for(int i = 0; i < n_idx; ++i) {
    if(clusters[i] != cluster) continue;
    hand_unindex(&indexer, round, i, cards);
    std::cout << cards_to_str(cards, card_sum) << "\n";
  }
}

void print_cluster(const int cluster, const int round, const int n_clusters) {
  const std::vector<int> clusters = cnpy::npy_load(cluster_file(round, n_clusters)).as_vec<int>();
  hand_indexer_t indexer;
  print_cluster(cluster, round, indexer, clusters);
}

void print_similar_boards(const std::string& board, const int n_clusters) {
  const int round = board.length() / 2 - 4;
  const std::vector<int> clusters = cnpy::npy_load(cluster_file(round, n_clusters)).as_vec<int>();
  hand_indexer_t indexer;

  uint8_t cards[7];
  str_to_cards(board, cards);
  const hand_index_t idx = hand_index_last(&indexer, cards);
  const int cluster = clusters[idx];
  std::cout << board << " cluster: " << cluster << std::endl;
  print_cluster(cluster, round, indexer, clusters);
}

}