#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iterator>
#include <sys/sysinfo.h>
#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/unordered_map.hpp>
#include <pluribus/cereal_ext.hpp>
#include <pluribus/util.hpp>
#include <pluribus/logging.hpp>
#include <pluribus/indexing.hpp>
#include <pluribus/mccfr.hpp>
#include <pluribus/range_viewer.hpp>
#include <pluribus/rng.hpp>
#include <pluribus/profiles.hpp>
#include <pluribus/blueprint.hpp>

namespace pluribus {

template<class T>
struct BlueprintBuffer {
  std::vector<std::pair<ActionHistory, std::vector<T>>> entries;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(entries);
  }

  void reset() {
    entries.clear();
  }
};

bool validate_preflop_fn(const std::string& preflop_fn, const std::vector<std::string>& all_fns) {
  return std::ranges::any_of(all_fns, [&preflop_fn](const std::string& fn) { return fn == preflop_fn; });
}

void set_meta_config(LosslessMetadata& meta, const TreeBlueprintSolver& bp) {
  meta.config = bp.get_config();
  meta.n_clusters = bp.get_strategy()->get_n_clusters();
  meta.tree_config = bp.get_strategy()->make_config_ptr();
  Logger::log("Initialized blueprint config:");
  Logger::log("n_clusters=" + std::to_string(meta.n_clusters));
  Logger::log("max_actions=" + std::to_string(meta.config.action_profile.max_actions()));
  Logger::log(meta.config.to_string());
}

template<class T>
std::vector<size_t> _collect_base_indexes(const TreeStorageNode<T>& strategy) {
  std::vector<size_t> base_idxs;
  base_idxs.reserve(strategy.history_map().size());
  for(const auto& entry : strategy.history_map()) {
    base_idxs.push_back(entry.second.idx);
  }
  base_idxs.push_back(strategy.data().size());
  std::ranges::sort(base_idxs);
  return base_idxs;
}

bool is_out_of_memory(const double free_gb, const double max_gb) {
  return get_free_ram() < static_cast<long long>(std::max(4.0, free_gb - max_gb) * pow(1024LL, 3LL));
}

template<class T>
void serialize_buffer(const std::string& buffer_prefix, BlueprintBuffer<T>& buffer, int& buf_idx, std::vector<std::string>& buffer_fns) {
  Logger::log("Saving buffer " + std::to_string(buf_idx) + "...");
  const std::string fn = buffer_prefix + std::to_string(buf_idx++) + ".bin";
  buffer_fns.push_back(fn);
  cereal_save(buffer, fn);
  Logger::log("Saved buffer " + std::to_string(buf_idx - 1) + " successfully.");
  buffer.reset();
}

void tree_to_lossless_buffers(const TreeStorageNode<int>* node, const double free_gb, const double max_gb, const std::filesystem::path& buffer_dir,
    const ActionHistory& history, BlueprintBuffer<float>& buffer, int& buf_idx, std::vector<std::string>& buffer_fns) {
  std::vector<float> values(node->get_n_values(), 0.0);
  for(int c = 0; c < node->get_n_clusters(); ++c) {
    const std::atomic<int>* base_ptr = node->get(c, 0);
    auto freq = calculate_strategy(base_ptr, node->get_value_actions().size());
    for(int a_idx = 0; a_idx < node->get_value_actions().size(); ++a_idx) {
      values[node_value_index(node->get_value_actions().size(), c, a_idx)] = freq[a_idx];
    }
  }
  buffer.entries.push_back({history, values});
  if(is_out_of_memory(free_gb, max_gb)) {
    serialize_buffer((buffer_dir / "lossless_buf_").string(), buffer, buf_idx, buffer_fns);
  }

  for(int a_idx = 0; a_idx < node->get_branching_actions().size(); ++a_idx) {
    if(node->is_allocated(a_idx)) {
      ActionHistory next_history = history;
      next_history.push_back(node->get_branching_actions()[a_idx]);
      tree_to_lossless_buffers(node->apply_index(a_idx), free_gb, max_gb, buffer_dir, next_history, buffer, buf_idx, buffer_fns);
    }
  }
}

LosslessMetadata build_lossless_buffers(const std::string& preflop_fn, const std::vector<std::string>& all_fns, const std::string& buf_dir, const double max_gb) {
  Logger::log("Building lossless buffers...");
  Logger::log("Preflop filename: " + preflop_fn);
  if(!validate_preflop_fn(preflop_fn, all_fns)) Logger::error("Preflop filename not found in all filenames.");

  std::ostringstream buf;
  LosslessMetadata meta;
  const std::filesystem::path buffer_dir = buf_dir;
  
  meta.preflop_buf_fn = (buffer_dir / "preflop_phi.bin").string();

  int buf_idx = 0;
  for(int bp_idx = 0; bp_idx < all_fns.size(); ++bp_idx) {
    Logger::log("Loading blueprint " + std::to_string(bp_idx) + "...");
    TreeBlueprintSolver bp;
    cereal_load(bp, all_fns[bp_idx]);
    const TreeStorageNode<int>* tree_root = bp.get_strategy();
    if(bp_idx == 0) {
      set_meta_config(meta, bp);
    }
    if(all_fns[bp_idx] == preflop_fn) {
      Logger::log("Found preflop blueprint. Storing phi...");
      cereal_save(*bp.get_phi(), meta.preflop_buf_fn);
    }

    const double free_gb = static_cast<double>(get_free_ram()) / pow(1024.0, 3.0);
    if(std::min(free_gb, max_gb) < 8) {
      Logger::error("At least 8G free RAM required to build blueprint. Available (G): " + std::to_string(free_gb));
    }
    BlueprintBuffer<float> buffer;
    Logger::log("Storing tree as buffers...");
    tree_to_lossless_buffers(tree_root, free_gb, max_gb, buffer_dir, meta.config.init_state.get_action_history(), buffer, buf_idx, meta.buffer_fns);
    if(!buffer.entries.empty()) {
      serialize_buffer((buffer_dir / "lossless_buf_").string(), buffer, buf_idx, meta.buffer_fns);
    }
  }
  Logger::log("Successfully built lossless buffers.");
  return meta;
}

LosslessMetadata collect_meta_data(const std::string& preflop_buf_fn, const std::string& final_bp_fn, const std::vector<std::string>& buffer_fns) {
  Logger::log("Collecting lossless meta data...");
  Logger::log("Preflop buffer file: " + preflop_buf_fn);
  Logger::log("Final blueprint file: " + final_bp_fn);
  LosslessMetadata meta;
  meta.preflop_buf_fn = preflop_buf_fn;
  meta.buffer_fns = buffer_fns;
  TreeBlueprintSolver final_bp;
  cereal_load(final_bp, final_bp_fn);
  set_meta_config(meta, final_bp);
  return meta;
}

void LosslessBlueprint::build(const std::string& preflop_fn, const std::vector<std::string>& all_fns, const std::string& buf_dir, const int max_gb) {
  Logger::log("Building lossless blueprint...");
  build_from_meta_data(build_lossless_buffers(preflop_fn, all_fns, buf_dir, max_gb));
}

void LosslessBlueprint::build_cached(const std::string& preflop_buf_fn, const std::string& final_bp_fn, const std::vector<std::string>& buffer_fns) {
  Logger::log("Building lossless blueprint from cached buffers...");
  build_from_meta_data(collect_meta_data(preflop_buf_fn, final_bp_fn, buffer_fns));
}

void set_preflop_strategy(TreeStorageNode<float>* node, const TreeStorageNode<float>* preflop_node, const PokerState& state) {
  if(state.get_round() > 0) return;
  if(node->get_n_values() != preflop_node->get_n_values()) {
    Logger::error("Preflop strategy size mismatch. Strategy values=" + std::to_string(node->get_n_values()) +
      ", Preflop values=" + std::to_string(node->get_n_values()));
  }
  if(node->get_branching_actions() != preflop_node->get_branching_actions()) {
    Logger::error("Preflop branching actions mismatch. Strategy actions=" + std::to_string(node->get_branching_actions().size()) +
      ", Preflop actions=" + std::to_string(node->get_branching_actions().size()));
  }
  for(int v_idx = 0; v_idx < preflop_node->get_n_values(); ++v_idx) {
    node->get_by_index(v_idx)->store(preflop_node->get_by_index(v_idx)->load());
  }
  for(int a_idx = 0; a_idx < preflop_node->get_branching_actions().size(); ++a_idx) {
    PokerState next_state = state.apply(preflop_node->get_branching_actions()[a_idx]);
    if(node->is_allocated(a_idx) != preflop_node->is_allocated(a_idx)) {
      if(next_state.get_round() == 0) {
        Logger::error("Preflop allocation mismatch for action " + preflop_node->get_branching_actions()[a_idx].to_string() + ".");
      }
    }
    else if(node->is_allocated(a_idx)) {
      set_preflop_strategy(node->apply_index(a_idx, next_state), preflop_node->apply_index(a_idx), next_state);
    }
  }
}

void normalize_tree(TreeStorageNode<float>* node, const PokerState& state) {
  for(int c = 0; c < node->get_n_clusters(); ++c) {
    std::atomic<float>* base_ptr = node->get(c, 0);
    auto freq = calculate_strategy(base_ptr, node->get_value_actions().size());
    for(int a_idx = 0; a_idx < node->get_value_actions().size(); ++a_idx) {
      base_ptr[a_idx].store(freq[a_idx]);
    }
  }
  for(int a_idx = 0; a_idx < node->get_branching_actions().size(); ++a_idx) {
    if(node->is_allocated(a_idx)) {
      PokerState next_state = state.apply(node->get_branching_actions()[a_idx]);
      normalize_tree(node->apply_index(a_idx, next_state), next_state);
    }
  }
}

void LosslessBlueprint::build_from_meta_data(const LosslessMetadata& meta) {
  Logger::log("Building lossless blueprint from meta data...");
  set_config(meta.config);
  assign_freq(new TreeStorageNode<float>{meta.config.init_state, meta.tree_config});
  for(std::string buf_fn : meta.buffer_fns) {
    BlueprintBuffer<float> buf;
    cereal_load(buf, buf_fn);
    Logger::log("Accumulating " + buf_fn + ": " + std::to_string(buf.entries.size()) + " nodes");
    #pragma omp parallel for schedule(static)
    for(size_t idx = 0; idx < buf.entries.size(); ++idx) {
      TreeStorageNode<float>* node = get_freq().get();
      PokerState state = meta.config.init_state;
      for(const Action a : buf.entries[idx].first.get_history()) {
        state = state.apply(a);
        node = node->apply(a, state);
      }
      if(node->get_n_values() != buf.entries[idx].second.size()) {
        Logger::error("Lossless buffer size mismatch. Buffer values=" + std::to_string(buf.entries[idx].second.size()) +
          ", Tree values=" + std::to_string(node->get_n_values()));
      }
      for(int v_idx = 0; v_idx < buf.entries[idx].second.size(); ++v_idx) {
        node->get_by_index(v_idx)->fetch_add(buf.entries[idx].second[v_idx]);
      }
    }
  }

  Logger::log("Setting preflop strategy to phi...");
  TreeStorageNode<float> phi;
  cereal_load(phi, meta.preflop_buf_fn);
  set_preflop_strategy(get_freq().get(), &phi, meta.config.init_state);

  Logger::log("Normalizing frequencies...");
  normalize_tree(get_freq().get(), meta.config.init_state);
  Logger::log("Lossless blueprint built.");
}

std::unordered_map<Action, uint8_t> build_compression_map(const ActionProfile& profile) {
  Logger::log("Building action compression map...");
  std::unordered_map<Action, uint8_t> compression_map;
  uint8_t idx = 0;
  for(Action a : profile.all_actions()) {
    Logger::log(a.to_string() + " -> " + std::to_string(idx));
    compression_map[a] = idx++;
  }
  return compression_map;
}

std::vector<Action> build_decompression_map(const std::unordered_map<Action, uint8_t>& compression_map) {
  Logger::log("Building action decompression map...");
  std::vector decompression_map(compression_map.size(), Action::UNDEFINED);
  for(const auto& [a, idx] : compression_map) {
    Logger::log(std::to_string(idx) + " -> " + a.to_string());
    decompression_map[idx] = a;
  }
  for(int i = 0; i < decompression_map.size(); ++i) {
    if(decompression_map[i] == Action::UNDEFINED) Logger::error("Unmapped compressed action idx: " + std::to_string(i));
  }
  return decompression_map;
}

std::vector<float> biased_freq(const std::vector<Action>& actions, const std::vector<float>& freq, const Action bias, const float factor) {
  std::vector<float> biased_freq;
  if(bias == Action::BIAS_FOLD || bias == Action::BIAS_CALL) {
    const Action biased_action = bias == Action::BIAS_FOLD ? Action::FOLD : Action::CHECK_CALL;
    if(const auto fold_it = std::ranges::find(actions, biased_action); fold_it != actions.end()) {
      const size_t fold_idx = std::distance(actions.begin(), fold_it);
      for(int fidx = 0; fidx < freq.size(); ++fidx) {
        biased_freq.push_back(fidx == fold_idx ? freq[fidx] * factor : freq[fidx]);
      }
    }
    else {
      biased_freq = freq;
    }
  }
  else if(bias == Action::BIAS_RAISE) {
    for(int fidx = 0; fidx < freq.size(); ++fidx) {
      biased_freq.push_back(actions[fidx].get_bet_type() > 0 || actions[fidx] == Action::ALL_IN ? freq[fidx] * factor : freq[fidx]);
    }
  }
  else if(bias == Action::BIAS_NONE) {
    biased_freq = freq;
  }
  else {
    Logger::error("Unknown bias: " + bias.to_string());
  }
  float sum = 0.0f;
  for(const float f : biased_freq) sum += f;
  for(int fidx = 0; fidx < biased_freq.size(); ++fidx) biased_freq[fidx] /= sum;
  return biased_freq;
}

Action sample_biased(const std::vector<Action>& actions, const std::vector<float>& freq, const Action bias, const float factor) {
  auto b_freq = biased_freq(actions, freq, bias, factor);
  std::discrete_distribution dist(b_freq.begin(), b_freq.end());
  return actions[dist(GlobalRNG::instance())];
}

void tree_to_sampled_buffers(const TreeStorageNode<float>* node, const double free_gb, const double max_gb, const std::filesystem::path& buffer_dir,
    const std::unordered_map<Action, uint8_t>& action_to_idx, const std::vector<Action>& biases, const float factor, const ActionHistory& history,
    BlueprintBuffer<uint8_t>& buffer, int& buf_idx, std::vector<std::string>& buffer_fns) {
  std::vector<uint8_t> sampled(node->get_n_clusters() * biases.size(), 0);
  for(int c = 0; c < node->get_n_clusters(); ++c) {
    const std::atomic<float>* base_ptr = node->get(c, 0);
    auto freq = calculate_strategy(base_ptr, node->get_value_actions().size());
    for(int a_idx = 0; a_idx < biases.size(); ++a_idx) {
      sampled[node_value_index(biases.size(), c, a_idx)] = action_to_idx.at(sample_biased(node->get_value_actions(), freq, biases[a_idx], factor));
    }
  }
  buffer.entries.push_back({history, sampled});
  if(is_out_of_memory(free_gb, max_gb)) {
    serialize_buffer((buffer_dir / "sampled_buf_").string(), buffer, buf_idx, buffer_fns);
  }

  for(int a_idx = 0; a_idx < node->get_branching_actions().size(); ++a_idx) {
    if(node->is_allocated(a_idx)) {
      ActionHistory next_history = history;
      next_history.push_back(node->get_branching_actions()[a_idx]);
      tree_to_sampled_buffers(node->apply_index(a_idx), free_gb, max_gb, buffer_dir, action_to_idx, biases, factor, next_history, buffer, buf_idx, buffer_fns);
    }
  }
}

SampledMetadata SampledBlueprint::build_sampled_buffers(const std::string& lossless_bp_fn, const std::string& buf_dir, const double max_gb,
    const ActionProfile& bias_profile, const float factor) {
  Logger::log("Building sampled buffers...");
  const std::filesystem::path buffer_dir = buf_dir;
  LosslessBlueprint bp;
  cereal_load(bp, lossless_bp_fn);
  SampledMetadata meta;
  meta.config = bp.get_config();
  meta.biases = bias_profile.get_actions(meta.config.init_state);
  Logger::log("Biases=" + std::to_string(meta.biases.size()));

  const auto action_to_idx = build_compression_map(bp.get_config().action_profile);
  _idx_to_action = build_decompression_map(action_to_idx);

  const double free_gb = static_cast<double>(get_free_ram()) / pow(1024.0, 3.0);
  if(std::min(free_gb, max_gb) < 8) {
    Logger::error("At least 8G free RAM required to build blueprint. Available (G): " + std::to_string(free_gb));
  }
  BlueprintBuffer<uint8_t> buffer;
  int buf_idx = 0;
  Logger::log("Storing tree as sampled buffers...");
  tree_to_sampled_buffers(bp.get_strategy(), free_gb, max_gb, buffer_dir, action_to_idx, meta.biases, factor, meta.config.init_state.get_action_history(), buffer, buf_idx, meta.buffer_fns);
  if(!buffer.entries.empty()) {
    serialize_buffer((buffer_dir / "sampled_buf_").string(), buffer, buf_idx, meta.buffer_fns);
  }
  Logger::log("Successfully built sampled buffers.");
  return meta;
}

std::unordered_map<Action, int> build_bias_offset_map(const PokerState& state, const ActionProfile& bias_profile) {
  Logger::log("Building bias offsets...");
  std::unordered_map<Action, int> bias_offset_map;
  for(const std::vector<Action>& all_biases = bias_profile.get_actions(state); const auto& bias : all_biases) {
    bias_offset_map[bias] = std::distance(all_biases.begin(), std::ranges::find(all_biases, bias));
    Logger::log(bias.to_string() + " -> " + std::to_string(bias_offset_map[bias]));
  }
  return bias_offset_map;
}

std::shared_ptr<const TreeStorageConfig> make_sampled_tree_config(const SampledMetadata& meta) {
  return std::make_shared<TreeStorageConfig>(TreeStorageConfig{
    meta.tree_config->cluster_spec,
    ActionMode::make_sampled_mode(meta.config.action_profile, meta.biases)
  });
}

void SampledBlueprint::build(const std::string& lossless_bp_fn, const std::string& final_snapshot_fn, const std::string& buf_dir, const int max_gb,
    const float bias_factor) {
  Logger::log("Building sampled blueprint...");
  const BiasActionProfile bias_profile;
  SampledMetadata meta = build_sampled_buffers(lossless_bp_fn, buf_dir, max_gb, bias_profile, bias_factor);
  {
    Logger::log("Loading tree config from final snapshot...");
    TreeBlueprintSolver final_snapshot;
    cereal_load(final_snapshot, final_snapshot_fn);
    meta.tree_config = final_snapshot.get_strategy()->make_config_ptr();
  }

  Logger::log("Initializing sampled blueprint...");
  assign_freq(new TreeStorageNode<uint8_t>(meta.config.init_state, make_sampled_tree_config(meta)));
  for(const auto& buf_fn : meta.buffer_fns) {
    BlueprintBuffer<uint8_t> buf;
    cereal_load(buf, buf_fn);
    Logger::log("Setting sampled actions from buffer " + buf_fn + ": " + std::to_string(buf.entries.size()) + " nodes");
    // #pragma omp parallel for schedule(static)
    for(size_t idx = 0; idx < buf.entries.size(); ++idx) {
      TreeStorageNode<uint8_t>* node = get_freq().get();
      PokerState state = meta.config.init_state;
      Logger::log("History: " + state.get_action_history().to_string());
      for(const Action a : buf.entries[idx].first.get_history()) {
        state = state.apply(a);
        node = node->apply(a, state);
      }
      Logger::log("Here");
      if(node->get_n_values() != buf.entries[idx].second.size()) {
        Logger::error("Sampled buffer size mismatch. Buffer values=" + std::to_string(buf.entries[idx].second.size()) +
          ", Tree values=" + std::to_string(node->get_n_values()));
      }
      Logger::log("Buffer values=" + std::to_string(buf.entries[idx].second.size()) + ", Tree values=" + std::to_string(node->get_n_values()));
      for(int v_idx = 0; v_idx < buf.entries[idx].second.size(); ++v_idx) {
        node->get_by_index(v_idx)->store(buf.entries[idx].second[v_idx]);
      }
    }
  }
  std::cout << "Sampled blueprint built.\n";
  _bias_to_offset = build_bias_offset_map(meta.config.init_state, bias_profile);
}

}
