#pragma once

#include <unordered_map>
#include <pluribus/poker.hpp>

namespace pluribus {

std::unordered_map<Hand, double> build_distribution(long n, const std::function<void(std::unordered_map<Hand, double>&)> &sampler, 
    bool verbose = true);
double distribution_rmse(const std::unordered_map<Hand, double>& dist_1, const std::unordered_map<Hand, double>& dist_2);
void print_distribution(const std::unordered_map<Hand, double>& dist);

}

