#pragma once

#include <vector>
#include <atomic>

namespace pluribus {

int sample_action_idx(const std::vector<float>& freq);

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
    const float uni = 1.0f / n_actions;
    for(auto& f : freq) {
      f = uni;
    }
  }
  return freq;
}

}
