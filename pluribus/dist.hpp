#pragma once

#include <unordered_map>
#include <pluribus/poker.hpp>

namespace pluribus {

PokerRange build_distribution(long n, const std::function<void(PokerRange&)> &sampler, bool verbose = true);
void distribution_to_png(const PokerRange& dist, const std::string& fn);
double distribution_rmse(const PokerRange& dist_1, const PokerRange& dist_2);

}

