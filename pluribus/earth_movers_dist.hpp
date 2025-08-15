#pragma once
#include <numeric>
#include <ranges>
#include <vector>
#include <pluribus/logging.hpp>

namespace pluribus {

template<size_t C>
double emd_heuristic(const std::vector<int>& x, const std::vector<double>& m, const std::array<std::vector<std::pair<double, int>>, C>& sorted_distances) {
  const size_t Q = m.size();
  for(const int c : x) if(c >= C) Logger::error("Cluster in x is too large: " + std::to_string(c));
  if(abs(std::accumulate(m.begin(), m.end(), 0.0) - 1.0) > 1e-6) Logger::error("m does not sum to 1.0.");
  for(const auto& vec : sorted_distances) {
    if(vec.size() != Q) Logger::error("Sorted distances vector size mismatch.");
    for(int idx = 0; idx < Q; ++idx) {
      if(idx > 0 && vec[idx - 1].first > vec[idx].first) {
        Logger::error("Distances are not sorted: " + std::to_string(vec[idx - 1].first) + " > " + std::to_string(vec[idx].first));
      }
      if(vec[idx].second >= m.size()) Logger::error("Ordered cluster index is out of bounds.");
    }
  }

  std::vector targets(x.size(), 1.0 / x.size());
  auto mean_remaining = m;
  std::vector done(x.size(), false);
  double tot_cost = 0.0;
  for(int i = 0; i < m.size(); ++i) {
    for(int j = 0; j < x.size(); ++j) {
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

void build_emd_matrix(const std::filesystem::path& dir);

}
