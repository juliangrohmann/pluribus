#pragma once

#include <string>
#include <vector>
#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>
#include <pluribus/debug.hpp>
#include <pluribus/rng.hpp>
#include <pluribus/sampling.hpp>
#include <pluribus/cereal_ext.hpp>
#include <pluribus/config.hpp>
#include <pluribus/tree_storage.hpp>

namespace pluribus {

struct LosslessMetadata {
  SolverConfig config;
  std::shared_ptr<const TreeStorageConfig> tree_config;
  std::vector<std::string> buffer_fns;
  std::string preflop_buf_fn;
  int n_clusters = -1;
};

template <class T>
class Blueprint : public Strategy<T> {
public:
  Blueprint() : _freq{nullptr} {}

  const TreeStorageNode<T>* get_strategy() const override {
    if(_freq) return _freq.get();
    throw std::runtime_error("Blueprint strategy is null.");
  }
  const SolverConfig& get_config() const override { return _config; }

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_freq, _config);
  }

protected:
  void assign_freq(TreeStorageNode<T>* freq) { _freq = std::unique_ptr<TreeStorageNode<T>>{freq}; }
  std::unique_ptr<TreeStorageNode<T>>& get_freq() { return _freq; }
  void set_config(const SolverConfig& config) { _config = config; }

private:
  std::unique_ptr<TreeStorageNode<T>> _freq;
  SolverConfig _config;
};

class LosslessBlueprint : public Blueprint<float> {
public:
  void build(const std::string& preflop_fn, const std::vector<std::string>& all_fns, const std::string& buf_dir = "", int max_gb = 50);
  void build_cached(const std::string& preflop_buf_fn, const std::string& final_bp_fn, const std::vector<std::string>& buffer_fns);

private:
  void build_from_meta_data(const LosslessMetadata& meta);
};

std::vector<float> biased_freq(const std::vector<Action>& actions, const std::vector<float>& freq, Action bias, float factor);
void _validate_ev_inputs(const PokerState& state, int i, const std::vector<PokerRange>& ranges, const std::vector<uint8_t>& board);

struct SampledMetadata {
  SolverConfig config;
  std::shared_ptr<const TreeStorageConfig> tree_config;
  std::vector<std::string> buffer_fns;
  std::vector<Action> biases;
};

class SampledBlueprint : public Blueprint<uint8_t> {
public:
  void build(const std::string& lossless_bp_fn, const std::string& final_snapshot_fn, const std::string& buf_dir, int max_gb = 50, float bias_factor = 5.0f);
  Action decompress_action(const uint8_t action_idx) const { return _idx_to_action[action_idx]; }
  int bias_offset(const Action bias) const { return _bias_to_offset.at(bias); }

private:
  SampledMetadata build_sampled_buffers(const std::string& lossless_bp_fn, const std::string& buf_dir, double max_gb, const ActionProfile& bias_profile,
    float factor);

  std::vector<Action> _idx_to_action;
  std::unordered_map<Action, int> _bias_to_offset;
};

}
