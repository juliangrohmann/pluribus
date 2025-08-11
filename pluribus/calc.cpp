#include <pluribus/calc.hpp>
#include <pluribus/rng.hpp>

namespace pluribus {

int sample_action_idx(const std::vector<float>& freq) {
  std::discrete_distribution dist(freq.begin(), freq.end());
  return dist(GlobalRNG::instance());
}

int sample_action_idx_fast(const std::vector<float>& freq) {
  float cumsum = 0.0f;
  const auto r = static_cast<float>(GSLGlobalRNG::uniform());
  for(int i = 0; i < static_cast<int>(freq.size()); ++i) {
    cumsum += freq[i];
    if(r <= cumsum) return i;
  }
  return static_cast<int>(freq.size()) - 1;
}

}
