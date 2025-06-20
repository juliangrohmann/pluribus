#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>
#include <pluribus/debug.hpp>
#include <pluribus/rng.hpp>
#include <pluribus/sampling.hpp>
#include <pluribus/cereal_ext.hpp>
#include <pluribus/storage.hpp>
#include <pluribus/mccfr.hpp>

namespace pluribus {

template <class T>
class Blueprint : public Strategy<T> {
public:
  Blueprint() : _freq{nullptr} {}

  const StrategyStorage<T>& get_strategy() const {
    if(_freq) return *_freq;
    throw std::runtime_error("Blueprint strategy is null.");
  }
  const BlueprintTrainerConfig& get_config() const { return _config; }
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
  double enumerate_ev(const PokerState& state, int i, const std::vector<PokerRange>& ranges, const std::vector<uint8_t>& board) const;

private:
  double node_ev(const PokerState& state, int i, const std::vector<Hand>& hands, const Board& board, int stack_size, std::vector<CachedIndexer>& indexers, const omp::HandEvaluator& eval) const;
};

std::vector<float> biased_freq(const std::vector<Action>& actions, const std::vector<float>& freq, Action bias, float factor);
void _validate_ev_inputs(const PokerState& state, int i, const std::vector<PokerRange>& ranges, const std::vector<uint8_t>& board);

class SampledBlueprint : public Blueprint<Action> {
public:
  void build(const std::string& lossless_bp_fn, const std::string& buf_dir, float bias_factor = 5.0f);
};

template<class T>
std::vector<size_t> _collect_base_indexes(const StrategyStorage<T>& strategy) {
  std::vector<size_t> base_idxs;
  base_idxs.reserve(strategy.history_map().size());
  for(const auto& entry : strategy.history_map()) {
    base_idxs.push_back(entry.second.idx);
  }
  base_idxs.push_back(strategy.data().size());
  std::sort(base_idxs.begin(), base_idxs.end());
  return base_idxs;
}

}
