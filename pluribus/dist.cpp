#include <pluribus/constants.hpp>
#include <pluribus/range.hpp>
#include <pluribus/dist.hpp>

namespace pluribus {

std::unordered_map<Hand, double> build_distribution(long n, std::function<void(std::unordered_map<Hand, double>&)> sampler, 
    bool verbose) {
  std::unordered_map<Hand, double> dist;
  long log_interval = n / 10;
  for(long i = 0; i < n; ++i) {
    if(verbose && i % log_interval == 0) {
      std::cout << std::fixed << std::setprecision(0) << "Build distribution: " << i / static_cast<double>(n) * 100.0 << "%\n";
    }
    sampler(dist);
  }
  double sum = 0.0;
  for(const auto& entry : dist) sum += entry.second;
  for(auto& entry : dist) entry.second /= sum;
  return dist;
}

void print_distribution(const std::unordered_map<Hand, double>& dist) {
  for(int i = 0; i < MAX_COMBOS; ++i) {
    Hand hand = HoleCardIndexer::get_instance()->hand(i);
    auto it = dist.find(hand);
    if(it != dist.end()) {
      std::cout << hand.to_string() << ": " << (*it).second * 100.0 << "%\n";
    }
  }
}

double distribution_rmse(const std::unordered_map<Hand, double>& dist_1, const std::unordered_map<Hand, double>& dist_2) {
  double rmse = 0.0;
  for(int i = 0; i < MAX_COMBOS; ++i) {
    Hand hand = HoleCardIndexer::get_instance()->hand(i);
    auto it_1 = dist_1.find(hand);
    auto it_2 = dist_2.find(hand);
    auto p_1 = it_1 != dist_1.end() ? (*it_1).second : 0.0;
    auto p_2 = it_2 != dist_2.end() ? (*it_2).second : 0.0;
    rmse += pow(p_1 - p_2, 2);
  }
  return pow(rmse, 0.5);
}

}
