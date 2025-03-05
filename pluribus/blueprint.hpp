#pragma once

#include <string>
#include <vector>
#include <cereal/cereal.hpp>
#include <pluribus/cereal_ext.hpp>
#include <pluribus/storage.hpp>

namespace pluribus {

class Blueprint {
public:
  Blueprint() : _freq{nullptr} {}

  void build(const std::string& preflop_fn, const std::vector<std::string>& postflop_fns, const std::string& buf_dir = "");

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_freq);
  }

private:
  std::unique_ptr<StrategyStorage<float>> _freq;
};

struct Buffer {
  tbb::concurrent_vector<int> regrets;
  size_t offset;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(regrets, offset);
  }
};

}
