#include <string>
#include <vector>
#include <cnpy.h>
#include <hand_isomorphism/hand_index.h>
#include <pluribus/util.hpp>
#include <pluribus/cluster.hpp>
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

}