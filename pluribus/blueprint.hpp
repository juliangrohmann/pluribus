#pragma once

#include <string>
#include <vector>
#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>
#include <pluribus/cereal_ext.hpp>
#include <pluribus/storage.hpp>
#include <pluribus/mccfr.hpp>

namespace pluribus {

template <class T>
class Blueprint : public Strategy<T> {
public:
  Blueprint() : _freq{nullptr} {}

  virtual const StrategyStorage<T>& get_strategy() const {
    if(_freq) return *_freq;
    throw std::runtime_error("Blueprint strategy is null.");
  }
  virtual const BlueprintTrainerConfig& get_config() const { return _config; }
  void set_config(BlueprintTrainerConfig config) { _config = config; }
  
  template <class Archive>
  void serialize(Archive& ar) {
    ar(_freq);
  }

protected:
  void assign_freq(StrategyStorage<T>* freq) { _freq = std::unique_ptr<StrategyStorage<T>>{freq}; }
  std::unique_ptr<StrategyStorage<T>>& get_freq() { return _freq; }

private:
  std::unique_ptr<StrategyStorage<T>> _freq;
  BlueprintTrainerConfig _config;
};

class LosslessBlueprint : public Blueprint<float> {
public:
  void build(const std::string& preflop_fn, const std::vector<std::string>& postflop_fns, const std::string& buf_dir = "");
  double enumerate_ev(const PokerState& state, int i, const PokerRange& hero, const PokerRange& villain, const std::vector<uint8_t>& board) const;

private:
  double expected_value(const PokerState& state, int i, const std::vector<Hand>& hands, const std::vector<uint8_t>& board, int stack_size, const omp::HandEvaluator& eval) const;
};

class SampledBlueprint : public Blueprint<Action> {
public:
  SampledBlueprint(const std::string& lossless_bp_fn, const std::string& buf_dir, float bias_factor = 5.0f);

  double monte_carlo_ev(const PokerState& state);
};

std::vector<float> biased_freq(const std::vector<Action>& actions, const std::vector<float>& freq, Action bias, float factor);

}
