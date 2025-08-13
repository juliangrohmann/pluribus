#include <pluribus/calc.hpp>

namespace pluribus {

int sample_action_idx(const float freq[], const int n_actions) {
  float cumsum = 0.0f;
  const auto r = static_cast<float>(GSLGlobalRNG::uniform());
  for(int i = 0; i < n_actions; ++i) {
    cumsum += freq[i];
    if(r <= cumsum) return i;
  }
  return n_actions - 1;
}

}
