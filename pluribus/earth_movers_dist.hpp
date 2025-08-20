#pragma once

#include <numeric>
#include <ranges>
#include <vector>
#include <cereal/types/vector.hpp>
#include <pluribus/cereal_ext.hpp>
#include <pluribus/logging.hpp>

namespace pluribus {

double emd_heuristic(const std::vector<int>& x, const std::vector<double>& x_w, const std::vector<double>& m_w,
    const std::vector<std::vector<std::pair<double, int>>>& sorted_distances);
void build_emd_preproc_cache(int start, int end, const std::filesystem::path& dir);

}
