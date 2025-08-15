#include <chrono>
#include <cnpy.h>
#include <pluribus/cereal_ext.hpp>
#include <pluribus/earth_movers_dist.hpp>
#include <pluribus/indexing.hpp>
#include <pluribus/logging.hpp>
#include <pluribus/poker.hpp>

namespace pluribus {

double emd_heuristic(const std::vector<int>& x, const std::vector<double>& x_w, const std::vector<double>& m_w,
    const std::vector<std::vector<std::pair<double, int>>>& sorted_distances) {
  const size_t C = sorted_distances.size();
  const size_t N = x.size();
  const size_t Q = m_w.size();

  for(const int c : x) if(c >= C) Logger::error("Cluster in x is too large: " + std::to_string(c));
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
    if(!(mask && card_mask(card))) {
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
      weights[weights.size() - 1] == unit;
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
  std::vector<std::vector<std::pair<double, int>>> sorted_distances;
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
    uint8_t cards[3];
    FlopIndexer::get_instance()->unindex(flop_idx, cards);
    Logger::log("Flop: " + cards_to_str(cards, 3));

    std::vector<hand_index_t> indexes;
    cereal_load(indexes, dir / ("indexes_r3_f" + std::to_string(flop_idx) + ".bin"));
    const std::vector<int> clusters = cnpy::npy_load(
      dir / ("clusters_r3_f" + std::to_string(flop_idx) + "_c" + std::to_string(n_clusters) + ".npy")).as_vec<int>();
    Logger::log("Building cluster map...");
    auto cluster_map = build_cluster_map(indexes, clusters);
    EMDPreprocCache cache;
    Logger::log("Building OCHS matrix...");
    cache.ochs_matrix = build_ochs_matrix(flop_idx, n_clusters, dir);
    Logger::log("Indexes: " + std::to_string(indexes.size()));
    const unsigned long log_interval = indexes.size() / 10UL;
    const auto t_0 = std::chrono::high_resolution_clock::now();
    for(hand_index_t i = 0; i < indexes.size(); ++i) {
      if(i > 0 && i % log_interval == 0) progress_str(i, indexes.size(), t_0);
      auto [unique_histogram, weights] = preprocess(build_histogram(indexes[i], cluster_map));
      std::cout << "Unique histogram: [" << join_as_strs(unique_histogram, " ") << "]\n";
      std::cout << std::fixed << std::setprecision(2) << "Weights: [" << join_as_strs(weights, " ") << "]\n";
      cache.histograms.push_back(unique_histogram);
      cache.weights.push_back(weights);
    }
    cereal_save(cache, dir / ("emd_preproc_cache_r2_f" + std::to_string(flop_idx) + "_c" + std::to_string(n_clusters) + ".bin"));
  }
}

}