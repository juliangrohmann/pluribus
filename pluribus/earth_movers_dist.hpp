#pragma once

#include <numeric>
#include <ranges>
#include <vector>
#include <cereal/types/vector.hpp>
#include <pluribus/cereal_ext.hpp>
#include <pluribus/logging.hpp>

namespace pluribus {

struct EMDPreprocCache {
  std::vector<std::vector<double>> ochs_matrix;
  std::vector<std::vector<int>> histograms;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(ochs_matrix, histograms);
  }
};

double emd_heuristic(const std::vector<int>& x, size_t Q, const std::vector<std::vector<std::pair<double, int>>>& sorted_distances);
void build_emd_preproc_cache(const std::filesystem::path& dir);

}
