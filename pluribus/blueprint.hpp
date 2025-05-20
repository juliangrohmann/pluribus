#pragma once

#include <string>
#include <vector>
#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>
#include <pluribus/cereal_ext.hpp>
#include <pluribus/storage.hpp>
#include <pluribus/mccfr.hpp>

namespace pluribus {

class Blueprint : public Strategy<float> {
public:
  Blueprint() : _freq{nullptr} {}

  void build(const std::string& preflop_fn, const std::vector<std::string>& postflop_fns, const std::string& buf_dir = "");

  virtual const StrategyStorage<float>& get_strategy() const {
    if(_freq) return *_freq;
    throw std::runtime_error("Blueprint strategy is null.");
  }
  virtual const BlueprintTrainerConfig& get_config() const { return _config; }
  
  template <class Archive>
  void serialize(Archive& ar) {
    ar(_freq);
  }

private:
  std::unique_ptr<StrategyStorage<float>> _freq;
  BlueprintTrainerConfig _config;
};

struct Buffer {
  std::vector<float> freqs;
  size_t offset;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(freqs, offset);
  }
};

}
