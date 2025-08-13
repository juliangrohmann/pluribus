#pragma once

#include <atomic>
#include <vector>
#include <pluribus/rng.hpp>

namespace pluribus {

int sample_action_idx(const float freq[], int n_actions);

template<class T>
inline int sample_idx_from_regrets(const std::atomic<T>* base_ptr, const int n_actions) {
  float w_local[16];
  float S = 0.f;
  for(int i = 0; i < n_actions; ++i) {
    const float v = static_cast<float>(base_ptr[i].load(std::memory_order_relaxed));
    const float w = v > 0.f ? v : 0.f;
    w_local[i] = w;
    S += w;
  }
  const float u01 = GSLGlobalRNG::uniform();
  if(S <= 0.f) {
    const int k = static_cast<int>(u01 * n_actions);
    return k < n_actions ? k : (n_actions-1);
  }
  const float threshold = u01 * S;
  float c = 0.f;
  for(int i=0;i<n_actions;++i) {
    c += w_local[i];
    if(c >= threshold) return i;
  }
  return n_actions-1;
}

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

template <class T>
void calculate_strategy_in_place(const std::atomic<T>* base_ptr, const int n_actions, float* buffer_ptr) {
  float sum = 0;
  for(int a_idx = 0; a_idx < n_actions; ++a_idx) {
    const float value = std::max(static_cast<float>(base_ptr[a_idx].load(std::memory_order_relaxed)), 0.0f);
    buffer_ptr[a_idx] = value;
    sum += value;
  }

  if(sum > 0) {
    for(int i = 0; i < n_actions; ++i) {
      buffer_ptr[i] /= sum;
    }
  }
  else {
    const float uni = 1.0f / static_cast<float>(n_actions);
    for(int i = 0; i < n_actions; ++i) {
      buffer_ptr[i] = uni;
    }
  }
}

}
