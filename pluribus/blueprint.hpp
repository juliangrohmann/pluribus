#pragma once

#include <string>
#include <vector>
#include <pluribus/storage.hpp>

namespace pluribus {

class Blueprint {
public:
  Blueprint() : _freq{nullptr} {}

  void distill(const std::string& preflop_fn, const std::vector<std::string>& postflop_fns);

private:
  std::unique_ptr<StrategyStorage<float>> _freq;
};

struct Buffer {
  std::vector<int> regrets;
  size_t offset;
};

}
