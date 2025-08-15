#include <chrono>
#include <cnpy.h>
#include <pluribus/cereal_ext.hpp>
#include <pluribus/earth_movers_dist.hpp>
#include <pluribus/indexing.hpp>
#include <pluribus/logging.hpp>
#include <pluribus/poker.hpp>

#include "cluster.hpp"

namespace pluribus {

void validate_clusters(const std::vector<int>& c_vec, const int C) {
  for(const int c : c_vec) if(c >= C) Logger::error("Cluster is too large: " + std::to_string(c));
}

void validate_weights(const std::vector<double>& w) {
  if(abs(std::accumulate(w.begin(), w.end(), 0.0) - 1.0) > 1e-6) Logger::error("Weights do not sum to 1.0.");
}

double emd_heuristic(const std::vector<int>& x, const std::vector<double>& x_w, const std::vector<double>& m_w,
    const std::vector<std::vector<std::pair<double, int>>>& sorted_distances) {
  const size_t C = sorted_distances.size();
  const size_t N = x.size();
  const size_t Q = m_w.size();

  validate_clusters(x, static_cast<int>(C));
  validate_weights(x_w);
  validate_weights(m_w);
  for(const auto& vec : sorted_distances) {
    if(vec.size() != Q) Logger::error("Sorted distances vector size mismatch.");
    for(int idx = 0; idx < Q; ++idx) {
      if(idx > 0 && vec[idx - 1].first > vec[idx].first) {
        Logger::error("Distances are not sorted: " + std::to_string(vec[idx - 1].first) + " > " + std::to_string(vec[idx].first));
      }
      if(vec[idx].second >= Q) Logger::error("Ordered cluster index is out of bounds.");
    }
  }

  std::vector targets = x_w;
  std::vector mean_remaining = m_w;
  std::vector done(N, false);
  double tot_cost = 0.0;
  for(int i = 0; i < Q; ++i) {
    for(int j = 0; j < N; ++j) {
      if(done[j]) continue;
      const int point_cluster = x[j];
      const int mean_cluster = sorted_distances[point_cluster][i].second;
      const double amt_remaining = mean_remaining[mean_cluster];
      if(amt_remaining == 0) continue;
      const double d = sorted_distances[point_cluster][i].first;
      if(amt_remaining < targets[j]) {
        tot_cost += amt_remaining * d;
        targets[j] -= amt_remaining;
        mean_remaining[mean_cluster] = 0;
      }
      else {
        tot_cost += targets[j] * d;
        mean_remaining[mean_cluster] -= targets[j];
        targets[j] = 0;
        done[j] = true;
      }
    }
  }
  return tot_cost;
}

std::vector<std::vector<double>> build_ochs_matrix(const hand_index_t flop_idx, const int n_clusters, const std::filesystem::path& dir) {
  constexpr int n_features = 8;
  const std::vector<float> centroids = cnpy::npy_load(
    dir / ("centroids_r3_f" + std::to_string(flop_idx) + "_c" + std::to_string(n_clusters) + ".npy")).as_vec<float>();
  if(centroids.size() != n_features * n_clusters) {
    Logger::error("Expected " + std::to_string(n_features * n_clusters) + "features. Got: " + std::to_string(centroids.size()));
  }
  auto matrix = std::vector(n_clusters, std::vector(n_clusters, 0.0));
  for(int c1 = 0; c1 < n_clusters; ++c1) {
    for(int c2 = c1 + 1; c2 < n_clusters; ++c2) {
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

std::vector<int> build_histogram(const hand_index_t turn_idx, const std::unordered_map<hand_index_t, int>& cluster_map) {
  constexpr int round = 2;
  uint8_t cards[7];
  HandIndexer::get_instance()->unindex(turn_idx, cards, round);
  const uint64_t mask = card_mask(cards, n_board_cards(round) + 2);
  std::vector<int> histogram;
  for(uint8_t card = 0; card < MAX_CARDS; ++card) {
    if(!(mask & card_mask(card))) {
      cards[6] = card;
      const hand_index_t river_idx = HandIndexer::get_instance()->index(cards, round + 1);
      histogram.push_back(cluster_map.at(river_idx));
    }
  }
  std::ranges::sort(histogram);
  return histogram;
}

std::pair<std::vector<int>, std::vector<double>> preprocess(const std::vector<int>& histogram) {
  std::vector<int> unique_histogram;
  std::vector<double> weights;
  const double unit = 1.0 / static_cast<double>(histogram.size());
  for(int i = 0; i < histogram.size(); ++i) {
    if(i > 0 && histogram[i] == histogram[i - 1]) {
      weights[weights.size() - 1] += unit;
    }
    else {
      weights.push_back(unit);
      unique_histogram.push_back(histogram[i]);
    }
  }
  return {unique_histogram, weights};
}

std::vector<std::vector<std::pair<double, int>>> build_sorted_distances(const std::vector<int>& mean_histogram, const std::vector<std::vector<double>>& ochs_matrix) {
  constexpr int n_clusters = 500;
  std::vector sorted_distances(n_clusters, std::vector<std::pair<double, int>>{});
  for(int c = 0; c < n_clusters; ++c) {
    for(int h_idx = 0; h_idx < mean_histogram.size(); ++h_idx) {
      sorted_distances[c].emplace_back(ochs_matrix[c][mean_histogram[h_idx]], h_idx);
    }
    std::ranges::sort(sorted_distances[c], [](const auto& e1, const auto& e2) { return e1.first < e2.first; });
  }
  return sorted_distances;
}

void build_emd_preproc_cache(const std::filesystem::path& dir) {
  constexpr int n_clusters = 500;
  Logger::log("Building EMD matrices...");
  for(hand_index_t flop_idx = 0; flop_idx < NUM_DISTINCT_FLOPS; ++flop_idx) {
    uint8_t cards[7];
    FlopIndexer::get_instance()->unindex(flop_idx, cards + 2);
    Logger::log("Flop: " + cards_to_str(cards + 2, 3));

    std::vector<hand_index_t> river_indexes;
    cereal_load(river_indexes, dir / ("indexes_r3_f" + std::to_string(flop_idx) + ".bin"));
    const std::vector<int> clusters = cnpy::npy_load(
      dir / ("clusters_r3_f" + std::to_string(flop_idx) + "_c" + std::to_string(n_clusters) + ".npy")).as_vec<int>();
    auto cluster_map = build_cluster_map(river_indexes, clusters);

    auto turn_index_set = collect_filtered_indexes(2, cards);
    auto turn_indexes = std::vector(turn_index_set.begin(), turn_index_set.end());
    cereal_save(turn_indexes, dir / ("indexes_r2_f" + std::to_string(flop_idx) + ".bin"));

    Logger::log("Building OCHS matrix...");
    auto ochs_matrix = build_ochs_matrix(flop_idx, n_clusters, dir);

    Logger::log("Preprocessing...");
    std::vector<std::vector<int>> histograms;
    std::vector<std::vector<double>> weights;
    for(const hand_index_t turn_idx : turn_indexes) {
      auto full_histogram = build_histogram(turn_idx, cluster_map);
      auto [unique_histogram, weight_vec] = preprocess(full_histogram);
      validate_clusters(unique_histogram, n_clusters);
      validate_weights(weight_vec);
      histograms.push_back(unique_histogram);
      weights.push_back(weight_vec);
    }

    Logger::log("Building EMD matrix...");
    auto matrix = std::vector(turn_indexes.size(), std::vector(turn_indexes.size(), 0.0));
    const unsigned long total_iter = (turn_indexes.size() * turn_indexes.size() - 1) / 2;
    const unsigned long log_interval = turn_indexes.size() / 100UL;
    const auto t_0 = std::chrono::high_resolution_clock::now();
    unsigned long iter = 0;
    for(hand_index_t idx1 = 0; idx1 < turn_indexes.size(); ++idx1) {
      for(hand_index_t idx2 = 0; idx2 < turn_indexes.size(); ++idx2) {
        if(iter > 0 && iter % log_interval == 0) Logger::log(progress_str(iter, total_iter, t_0));
        matrix[idx1][idx2] = emd_heuristic(histograms[idx1], weights[idx1], weights[idx2], build_sorted_distances(histograms[idx2], ochs_matrix));
        ++iter;
      }
    }
    cereal_save(matrix, dir / ("emd_matrix_r2_f" + std::to_string(flop_idx) + "_c" + std::to_string(n_clusters) + ".bin"));
  }
}

}
