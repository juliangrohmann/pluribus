#include <iostream>
#include <random>
#include <thread>

class GlobalRNG {
public:
  static std::mt19937& instance() {
    // thread_local std::mt19937 rng(std::random_device{}());
    thread_local std::mt19937 rng(1338);
    return rng;
  }
};
