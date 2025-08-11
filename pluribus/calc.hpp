#pragma once

#include <atomic>
#include <vector>

namespace pluribus {

int sample_action_idx(const std::vector<float>& freq);
int sample_action_idx_fast(const std::vector<float>& freq);

template <class T>
std::vector<float> calculate_strategy(const std::atomic<T>* base_ptr, const int n_actions) {
  std::vector<float> freq;
  freq.reserve(n_actions);
  float sum = 0;
  for(int a_idx = 0; a_idx < n_actions; ++a_idx) {
    float value = std::max(static_cast<float>(base_ptr[a_idx].load(std::memory_order_relaxed)), 0.0f);
    freq.push_back(value);
    sum += value;
  }

  if(sum > 0) {
    for(auto& f : freq) {
      f /= sum;
    }
  }
  else {
    const float uni = 1.0f / static_cast<float>(n_actions);
    for(auto& f : freq) {
      f = uni;
    }
  }
  return freq;
}

}
