#include <pluribus/calc.hpp>
#include <pluribus/rng.hpp>

namespace pluribus {

int sample_action_idx(const std::vector<float>& freq) {
  std::discrete_distribution dist(freq.begin(), freq.end());
  return dist(GlobalRNG::instance());
}

}