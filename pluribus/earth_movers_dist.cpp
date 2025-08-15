#include <chrono>
#include <cnpy.h>
#include <pluribus/cereal_ext.hpp>
#include <pluribus/earth_movers_dist.hpp>
#include <pluribus/indexing.hpp>
#include <pluribus/logging.hpp>
#include <pluribus/poker.hpp>

namespace pluribus {

std::vector<std::vector<double>> build_ochs_matrix(const hand_index_t flop_idx, const int n_clusters, const std::filesystem::path& dir) {
  constexpr int n_features = 8;
  const std::vector<float> centroids = cnpy::npy_load(
    dir / ("centroids_r3_f" + std::to_string(flop_idx) + "_c" + std::to_string(n_clusters) + ".npy")).as_vec<float>();
  if(centroids.size() != n_features * n_clusters) {
    Logger::error("Expected " + std::to_string(n_features * n_clusters) + "features. Got: " + std::to_string(centroids.size()));
  }
  auto matrix = std::vector(n_clusters, std::vector(n_clusters, 0.0));
  for(int c1 = 0; c1 < n_clusters; ++c1) {
    for(int c2 = c1 + 1; c1 < n_clusters; ++c2) {
      double dist = 0.0f;
      for(int i = 0; i < n_features; ++i) {
        dist += pow(centroids[c1 * n_features + i] - centroids[c2 * n_features + i], 2.0f);
      }
      dist = sqrt(dist);
      matrix[c1][c2] = dist;
      matrix[c2][c1] = dist;
    }
  }
  return matrix;
}

std::unordered_map<hand_index_t, int> build_cluster_map(const std::vector<hand_index_t>& indexes, const std::vector<int>& clusters) {
  if(indexes.size() != clusters.size()) {
    Logger::error("Indexes to clusters size mismatch: Indexes size=" + std::to_string(indexes.size()) + ", Clusters size=" + std::to_string(clusters.size()));
  }
  std::unordered_map<hand_index_t, int> cluster_map;
  for(int i = 0; i < indexes.size(); ++i) {
    cluster_map[indexes[i]] = clusters[i];
  }
  return cluster_map;
}

std::vector<int> build_histogram(const hand_index_t turn_idx, const std::unordered_map<hand_index_t, int>& clusters) {
  constexpr int round = 2;
  uint8_t cards[7];
  HandIndexer::get_instance()->unindex(turn_idx, cards, round);
  const uint64_t mask = card_mask(cards, n_board_cards(round) + 2);
  std::vector<int> histogram;
  for(uint8_t card = 0; card < MAX_CARDS; ++card) {
    if(!(mask && card_mask(card))) {
      cards[6] = card;
      const hand_index_t river_idx = HandIndexer::get_instance()->index(cards, round + 1);
      histogram.push_back(clusters.at(river_idx));
    }
  }
  std::ranges::sort(histogram);
  return histogram;
}

double compute_emd(const std::vector<int>& histogram1, const std::vector<int>& histogram2, const std::vector<std::vector<double>>& ochs_matrix) {
  constexpr int n_clusters = 500;
  std::array<std::vector<std::pair<double, int>>, n_clusters> sorted_distances;
  for(int c = 0; c < n_clusters; ++c) {
    for(int h_idx = 0; h_idx < histogram2.size(); ++h_idx) {
      sorted_distances[c].emplace_back(ochs_matrix[c][histogram2[h_idx]], h_idx);
    }
  }
  std::vector m(histogram2.size(), 1.0 / static_cast<double>(histogram2.size()));
  return emd_heuristic(histogram1, m, sorted_distances);
}

double symmetric_emd(const hand_index_t idx1, const hand_index_t idx2, const std::unordered_map<hand_index_t, int>& cluster_map,
    const std::vector<std::vector<double>>& ochs_matrix) {
  const auto histogram1 = build_histogram(idx1, cluster_map);
  const auto histogram2 = build_histogram(idx2, cluster_map);
  return 0.5 * (compute_emd(histogram1, histogram2, ochs_matrix) + compute_emd(histogram2, histogram1, ochs_matrix));
}

void build_emd_matrix(const std::filesystem::path& dir) {
  constexpr int n_clusters = 500;
  Logger::log("Building EMD matrices...");
  for(hand_index_t flop_idx = 0; flop_idx < NUM_DISTINCT_FLOPS; ++flop_idx) {
    uint8_t cards[3];
    FlopIndexer::get_instance()->unindex(flop_idx, cards);
    Logger::log("Flop: " + cards_to_str(cards, 3));

    std::vector<hand_index_t> indexes;
    cereal_load(indexes, dir / ("indexes_r3_f" + std::to_string(flop_idx) + ".bin"));
    const std::vector<int> clusters = cnpy::npy_load(
      dir / ("clusters_r3_f" + std::to_string(flop_idx) + "_c" + std::to_string(n_clusters) + ".npy")).as_vec<int>();
    auto cluster_map = build_cluster_map(indexes, clusters);
    auto ochs_matrix = build_ochs_matrix(flop_idx, n_clusters, dir);
    auto emd_matrix = std::vector(indexes.size(), std::vector(indexes.size(), 0.0));
    Logger::log("Indexes: " + std::to_string(indexes.size()));
    const long total_iters = indexes.size() * (indexes.size() - 1) / 2;
    const long log_interval = total_iters / 100;
    Logger::log("Iterations: " + std::to_string(total_iters));
    long iter = 0;
    const auto t_0 = std::chrono::high_resolution_clock::now();
    for(hand_index_t i = 0; i < indexes.size(); ++i) {
      for(hand_index_t j = i + 1; j < indexes.size(); ++j) {
        if(i > 0 && i % log_interval == 0) progress_str(i, total_iters, t_0);
        const double symm_emd = symmetric_emd(indexes[i], indexes[j], cluster_map, ochs_matrix);
        emd_matrix[i][j] = symm_emd;
        emd_matrix[j][i] = symm_emd;
        ++iter;
      }
    }
  }
}

}