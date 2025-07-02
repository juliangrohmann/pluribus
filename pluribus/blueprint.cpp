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
#include <pluribus/blueprint.hpp>

namespace pluribus {

struct LosslessBuffer {
  std::vector<float> freqs;
  size_t offset;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(freqs, offset);
  }
};

bool validate_preflop_fn(const std::string& preflop_fn, const std::vector<std::string>& all_fns) {
  for(const auto& fn : all_fns) {
    if(fn == preflop_fn) return true;
  }
  return false;
}

void set_meta_config(LosslessMetadata& meta, const MappedBlueprintSolver& bp) {
  meta.config = bp.get_config();
  meta.n_clusters = bp.get_strategy().n_clusters();
  Logger::log("Initialized blueprint config:");
  Logger::log("n_clusters=" + std::to_string(meta.n_clusters));
  Logger::log("max_actions=" + std::to_string(meta.config.action_profile.max_actions()));
  Logger::log(meta.config.to_string());
}

void set_meta_strategy_info(LosslessMetadata& meta, const StrategyStorage<int>& regrets) {
  meta.max_regrets = regrets.data().size();
  Logger::log("New max regrets: " + std::to_string(meta.max_regrets));
  meta.history_map = regrets.history_map();
  Logger::log("New history map size: " + std::to_string(meta.history_map.size()));
}

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

LosslessMetadata build_lossless_buffers(const std::string& preflop_fn, const std::vector<std::string>& all_fns, const std::string& buf_dir, int max_gb) {
  Logger::log("Building lossless buffers...");
  Logger::log("Preflop filename: " + preflop_fn);
  if(!validate_preflop_fn(preflop_fn, all_fns)) Logger::error("Preflop filename not found in all filenames.");

  std::ostringstream buf;
  LosslessMetadata meta;
  std::filesystem::path buffer_dir = buf_dir;
  
  meta.preflop_buf_fn = (buffer_dir / "preflop_phi.bin").string();

  int buf_idx = 0;
  for(int bp_idx = 0; bp_idx < all_fns.size(); ++bp_idx) {
    Logger::log("Loading blueprint " + std::to_string(bp_idx) + "...");
    MappedBlueprintSolver bp;
    cereal_load(bp, all_fns[bp_idx]);
    const auto& regrets = bp.get_strategy();
    if(bp_idx == 0) {
      set_meta_config(meta, bp);
    }
    if(regrets.data().size() > meta.max_regrets) {
      set_meta_strategy_info(meta, regrets);
    }
    if(all_fns[bp_idx] == preflop_fn) {
      Logger::log("Found preflop blueprint. Storing phi...");
      cereal_save(bp.get_phi(), meta.preflop_buf_fn);
    }

    std::vector<size_t> base_idxs = _collect_base_indexes(regrets);
    size_t free_ram = std::min(get_free_ram(), max_gb * 1000LL * 1000LL * 1000LL);
    if(free_ram < 8 * pow(1024, 3)) {
      Logger::error("At least 8G free RAM required to build blueprint. Available (bytes): " + std::to_string(free_ram));
    }
    size_t buf_sz = static_cast<size_t>((free_ram - 8 * pow(1024, 3)) / sizeof(float));
    Logger::log("Buffer element cutoff: " + std::to_string(buf_sz));

    size_t bidx_start = 0;
    size_t bidx_end = 0;
    while(bidx_start < base_idxs.size() - 1) {
      for(; bidx_end < base_idxs.size() - 1; ++bidx_end) {
        if(base_idxs[bidx_end] - base_idxs[bidx_start] > buf_sz) break;
      }
      if(bidx_end == bidx_start) Logger::error("Failed to increase bidx_end");

      LosslessBuffer buffer;
      buffer.offset = base_idxs[bidx_start];
      buffer.freqs.resize(base_idxs[bidx_end] - base_idxs[bidx_start]);

      Logger::log("Storing buffer: base_idxs[" + std::to_string(bidx_start) + ", " + std::to_string(bidx_end) + "), indeces: [" 
                + std::to_string(base_idxs[bidx_start]) + ", " + std::to_string(base_idxs[bidx_end]) + ")");
      #pragma omp parallel for schedule(dynamic)
      for(size_t curr_idx = bidx_start; curr_idx < bidx_end; ++curr_idx) {
        if(curr_idx + 1 >= base_idxs.size()) Logger::error("Buffering: Indexing base indeces out of range!");
        size_t n_entries = base_idxs[curr_idx + 1] - base_idxs[curr_idx];
        if(n_entries % meta.n_clusters != 0) Logger::error("Buffering: Indivisible regret section!");
        size_t n_actions = n_entries / meta.n_clusters;
        if(n_actions > meta.config.action_profile.max_actions()) Logger::error("Buffering: Too many actions in storage section:" + 
            std::to_string(n_actions) + " > " + std::to_string(meta.config.action_profile.max_actions()));

        for(int c = 0; c < meta.n_clusters; ++c) {
          size_t base_idx = base_idxs[curr_idx] + c * n_actions;
          auto freq = calculate_strategy(&regrets[base_idx], n_actions);
          for(int fidx = 0; fidx < freq.size(); ++fidx) {
            buffer.freqs[base_idx - base_idxs[bidx_start] + fidx] = freq[fidx];
          }
        }
      }
      bidx_start = bidx_end;

      Logger::log("Saving buffer " + std::to_string(buf_idx) + "...");
      std::string fn = (buffer_dir / ("lossless_buf_" + std::to_string(buf_idx++) + ".bin")).string();
      meta.buffer_fns.push_back(fn);
      cereal_save(buffer, fn);
      Logger::log("Saved buffer " + std::to_string(buf_idx - 1) + " successfully.");
    }
  }
  return meta;
}

LosslessMetadata collect_meta_data(const std::string& preflop_buf_fn, const std::string& final_bp_fn, const std::vector<std::string>& buffer_fns) {
  Logger::log("Collecting lossless meta data...");
  Logger::log("Preflop buffer file: " + preflop_buf_fn);
  Logger::log("Final blueprint file: " + final_bp_fn);
  LosslessMetadata meta;
  meta.preflop_buf_fn = preflop_buf_fn;
  meta.buffer_fns = buffer_fns;
  MappedBlueprintSolver final_bp;
  cereal_load(final_bp, final_bp_fn);
  set_meta_config(meta, final_bp);
  set_meta_strategy_info(meta, final_bp.get_strategy());
  return meta;
}

void LosslessBlueprint::build(const std::string& preflop_fn, const std::vector<std::string>& all_fns, const std::string& buf_dir, int max_gb) {
  Logger::log("Building lossless blueprint...");
  build_from_meta_data(build_lossless_buffers(preflop_fn, all_fns, buf_dir, max_gb));
}

void LosslessBlueprint::build_cached(const std::string& preflop_buf_fn, const std::string& final_bp_fn, const std::vector<std::string>& buffer_fns) {
  Logger::log("Building lossless blueprint from cached buffers...");
  build_from_meta_data(collect_meta_data(preflop_buf_fn, final_bp_fn, buffer_fns));
}

void LosslessBlueprint::build_from_meta_data(const LosslessMetadata& meta) {
  set_config(meta.config);
  assign_freq(new StrategyStorage<float>{meta.config.action_profile, meta.n_clusters});
  get_freq()->allocate(meta.max_regrets);
  for(std::string buf_fn : meta.buffer_fns) {
    LosslessBuffer buf;
    cereal_load(buf, buf_fn);
    Logger::log("Accumulating " + buf_fn + ": [" + std::to_string(buf.offset) + ", " + std::to_string(buf.offset + buf.freqs.size()) + ")");
    #pragma omp parallel for schedule(static)
    for(size_t idx = 0; idx < buf.freqs.size(); ++idx) {
      auto& entry = get_freq()->operator[](buf.offset + idx);
      entry.fetch_add(buf.freqs[idx]);
    }
  }

  Logger::log("Inserting histories...");
  for(const auto& entry : meta.history_map) {
    get_freq()->history_map()[entry.first] = entry.second;
  }

  Logger::log("Setting preflop strategy to phi...");
  StrategyStorage<float> phi;
  cereal_load(phi, meta.preflop_buf_fn);
  for(auto entry : phi.history_map()) {
    PokerState state = meta.config.init_state;
    state.apply(entry.first);
    int n_actions = valid_actions(state, meta.config.action_profile).size();
    for(int c = 0; c < phi.n_clusters(); ++c) {
      size_t phi_base_idx = phi.index(state, c);
      size_t freq_base_idx = get_freq()->index(state, c);
      for(int a_idx = 0; a_idx < n_actions; ++a_idx) {
        get_freq()->operator[](freq_base_idx + a_idx).store(phi[phi_base_idx + a_idx].load());
      }
    }
  }

  Logger::log("Normalizing frequencies...");
  std::vector<size_t> base_idxs = _collect_base_indexes(*get_freq());
  for(size_t curr_idx = 0; curr_idx < base_idxs.size() - 1; ++curr_idx) {
    size_t n_entries = base_idxs[curr_idx + 1] - base_idxs[curr_idx];
    if(n_entries % meta.n_clusters != 0) Logger::error("Renorm: Indivisible storage section!");
    size_t n_actions = n_entries / meta.n_clusters;
    if(n_actions > get_config().action_profile.max_actions()) Logger::error("Renorm: Too many actions in storage section!");
    for(int c = 0; c < meta.n_clusters; ++c) {
      size_t base_idx = base_idxs[curr_idx] + c * n_actions;
      float total = 0.0f;
      for(size_t aidx = 0; aidx < n_actions; ++aidx) {
        total += get_freq()->operator[](base_idx + aidx);
      }
      if(total > 0) {
        for(size_t aidx = 0; aidx < n_actions; ++aidx) {
          auto& entry = get_freq()->operator[](base_idx + aidx);
          entry.store(entry.load() / total);
        }
      }
      else {
        for(size_t aidx = 0; aidx < n_actions; ++aidx) {
          get_freq()->operator[](base_idx + aidx).store(1.0 / n_actions);
        }
      }
    }
  }
  Logger::log("Lossless blueprint built.");
}

std::unordered_map<Action, uint8_t> build_compression_map(const ActionProfile& profile) {
  Logger::log("Building action compression map...");
  std::unordered_map<Action, uint8_t> compression_map;
  uint8_t idx = 0;
  for(Action a : profile.all_actions()) {
    Logger::log(a.to_string() + " -> " + std::to_string(static_cast<int>(idx)));
    compression_map[a] = idx++;
  }
  return compression_map;
}

std::vector<Action> build_decompression_map(std::unordered_map<Action, uint8_t> compression_map) {
  Logger::log("Building action decompression map...");
  std::vector<Action> decompression_map(compression_map.size(), Action::UNDEFINED);
  for(const auto& entry : compression_map) {
    Logger::log(std::to_string(static_cast<int>(entry.second)) + " -> " + entry.first.to_string());
    decompression_map[entry.second] = entry.first;
  }
  for(int i = 0; i < decompression_map.size(); ++i) {
    if(decompression_map[i] == Action::UNDEFINED) Logger::error("Unmapped compressed action idx: " + std::to_string(i));
  }
  return decompression_map;
}

std::vector<float> biased_freq(const std::vector<Action>& actions, const std::vector<float>& freq, Action bias, float factor) {
  std::vector<float> biased_freq;
  if(bias == Action::BIAS_FOLD || bias == Action::BIAS_CALL) {
    Action biased_action = bias == Action::BIAS_FOLD ? Action::FOLD : Action::CHECK_CALL;
    auto fold_it = std::find(actions.begin(), actions.end(), biased_action);
    if(fold_it != actions.end()) {
      size_t fold_idx = std::distance(actions.begin(), fold_it);
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
  for(float f : biased_freq) sum += f;
  for(int fidx = 0; fidx < biased_freq.size(); ++fidx) biased_freq[fidx] /= sum;
  return biased_freq;
}

Action sample_biased(const std::vector<Action>& actions, const std::vector<float>& freq, Action bias, float factor) {
  auto b_freq = biased_freq(actions, freq, bias, factor);
  std::discrete_distribution<> dist(b_freq.begin(), b_freq.end());
  return actions[dist(GlobalRNG::instance())];
}

SampledMetadata SampledBlueprint::build_sampled_buffers(const std::string& lossless_bp_fn, const std::string& buf_dir, 
    const ActionProfile& bias_profile, float factor) {
  std::cout << "Building sampled buffers...\n";
  SampledMetadata meta;
  std::filesystem::path buffer_dir = buf_dir;
  LosslessBlueprint bp;
  cereal_load(bp, lossless_bp_fn);
  meta.config = bp.get_config();
  meta.biases = bias_profile.get_actions(0, 0, 0, 150);
  meta.n_clusters = bp.get_strategy().n_clusters();
  auto action_to_idx = build_compression_map(bp.get_config().action_profile);
  _idx_to_action = build_decompression_map(action_to_idx);

  std::cout << "Collecting histories... " << std::flush;
  meta.histories.reserve(bp.get_strategy().history_map().size());
  // size_t fold_saved = 0, bet_saved = 0;;
  PokerState init_state{meta.config.poker};
  for(const auto& entry : bp.get_strategy().history_map()) {
    PokerState state = init_state.apply(entry.first);
    if(!state.is_terminal()) {
      meta.histories.push_back(entry.first);
    }
    else {
      Logger::error("Found terminal state.");
    }
    // auto actions = valid_actions(state, bp.get_config().action_profile);
    // int bets = 0;
    // for(Action a : actions) {
    //   if(a.get_bet_type() > 0 || a == Action::ALL_IN) ++bets;
    // }
    // if(std::find(actions.begin(), actions.end(), Action::FOLD) == actions.end()) {
    //   fold_saved += 200;
    // }
    // if(bets == 0) {
    //   bet_saved += 200;
    // }
  }
  // std::cout << "Fold saved: " << fold_saved << '\n';
  // std::cout << "Bet saved: " << bet_saved << '\n';
  std::cout << "Collected " << meta.histories.size() << " (" << meta.n_clusters << " clusters).\n";

  std::cout << "Clusters=" << meta.n_clusters << ", Biases=" << meta.biases.size() << "\n";
  std::cout << "Building buffers...\n";
  std::unordered_map<ActionHistory, std::vector<uint8_t>> buffer;
  long long min_ram = 4 * pow(1024, 3);
  int buf_idx = 0;
  for(size_t hidx = 0; hidx < meta.histories.size(); ++hidx) {
    std::vector<uint8_t> sampled;
    sampled.reserve(meta.n_clusters * meta.biases.size());
    for(int c = 0; c < meta.n_clusters; ++c) {
      PokerState state = init_state.apply(meta.histories[hidx]);
      auto actions = valid_actions(state, bp.get_config().action_profile);
      size_t base_idx = bp.get_strategy().index(state, c);
      auto freq = calculate_strategy(&bp.get_strategy()[base_idx], actions.size());
      for(Action bias : meta.biases) {
        sampled.push_back(action_to_idx[sample_biased(actions, freq, bias, factor)]);
      }
    }
    buffer[meta.histories[hidx]] = sampled;

    if(get_free_ram() < min_ram || hidx == meta.histories.size() - 1) {
      std::cout << "Buffered " << buffer.size() << " histories.\n";
      if(buffer.size() == 0) Logger::error("Out of RAM but buffer is empty.");
      std::string fn = "sampled_buf_" + std::to_string(buf_idx++) + ".bin";
      meta.buffer_fns.push_back(fn);
      cereal_save(buffer, (buffer_dir / fn).string());
      buffer.clear();
    }
  }
  std::cout << "Buffers built.\n";
  return meta;
}

std::unordered_map<Action, int> build_bias_offset_map(const ActionProfile& bias_profile) {
  Logger::log("Building bias offsets...");
  std::unordered_map<Action, int> bias_offset_map;
  const std::vector<Action>& all_biases = bias_profile.get_actions(0, 0, 0, 0);
  for(const auto& bias : all_biases) {
    bias_offset_map[bias] = std::distance(all_biases.begin(), std::find(all_biases.begin(), all_biases.end(), bias));
    Logger::log(bias.to_string() + " -> " + std::to_string(bias_offset_map[bias]));
  }
  return bias_offset_map;
}

void SampledBlueprint::build(const std::string& lossless_bp_fn, const std::string& buf_dir, float bias_factor) {
  Logger::log("Building sampled blueprint...");
  BiasActionProfile bias_profile;
  SampledMetadata meta = build_sampled_buffers(lossless_bp_fn, buf_dir, bias_profile, bias_factor);
  std::filesystem::path buffer_dir = buf_dir;
  assign_freq(new StrategyStorage<Action>(bias_profile, meta.n_clusters));
  for(const auto& fn : meta.buffer_fns) {
    std::unordered_map<ActionHistory, std::vector<Action>> buffer;
    cereal_load(buffer, (buffer_dir / fn).string());
    for(size_t hidx = 0; hidx < meta.histories.size(); ++hidx) {
      PokerState state{meta.config.poker};
      state = state.apply(meta.histories[hidx]);
      size_t base_idx = get_freq()->index(state, 0);
      auto sampled = buffer[meta.histories[hidx]];
      for(int idx = 0; idx < sampled.size(); ++idx) {
        get_freq()->operator[](base_idx + idx) = sampled[idx];
      }
    }
  }
  std::cout << "Stored histories: " << get_freq()->history_map().size() << "\n";
  std::cout << "Data size: " << get_freq()->data().size() << "\n";
  std::cout << "Sampled blueprint built.\n";
  _bias_to_offset = build_bias_offset_map(bias_profile);
}

}
