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
#include <pluribus/debug.hpp>
#include <pluribus/logging.hpp>
#include <pluribus/indexing.hpp>
#include <pluribus/mccfr.hpp>
#include <pluribus/range_viewer.hpp>
#include <pluribus/rng.hpp>
#include <pluribus/blueprint.hpp>

namespace pluribus {

struct LosslessMetadata {
  BlueprintTrainerConfig config;
  tbb::concurrent_unordered_map<ActionHistory, HistoryEntry> history_map;
  std::vector<std::string> buffer_fns;
  size_t max_regrets = 0;
  int n_clusters = -1;
};

struct LosslessBuffer {
  std::vector<float> freqs;
  size_t offset;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(freqs, offset);
  }
};

LosslessMetadata build_lossless_buffers(const std::string& preflop_fn, const std::vector<std::string>& postflop_fns, const std::string& buf_dir) {
  Logger::log("Building lossless buffers...");
  std::ostringstream buf;
  LosslessMetadata meta;
  std::filesystem::path buffer_dir = buf_dir;
  
  int buf_idx = 0;
  for(int bp_idx = 0; bp_idx < postflop_fns.size(); ++bp_idx) {
    Logger::log("Loading blueprint " + std::to_string(bp_idx) + "...");
    BlueprintTrainer bp;
    cereal_load(bp, postflop_fns[bp_idx]);
    Logger::log("Loaded blueprint " + std::to_string(bp_idx) + " successfully.");
    const auto& regrets = bp.get_strategy();
    if(bp_idx == 0) {
      meta.config = bp.get_config();
      meta.n_clusters = regrets.n_clusters();
      Logger::log("Initialized blueprint config:");
      Logger::log("n_clusters=" + std::to_string(meta.n_clusters));
      Logger::log("max_actions=" + std::to_string(meta.config.action_profile.max_actions()));
      Logger::log(meta.config.to_string());
    }

    std::vector<size_t> base_idxs = _collect_base_indexes(regrets);
    size_t free_ram = get_free_ram();
    if(free_ram < 8 * pow(1024, 3)) {
      Logger::error("At least 8G free RAM required to build blueprint. Available (bytes): " + std::to_string(free_ram));
    }
    size_t buf_sz = static_cast<size_t>((free_ram - 8 * pow(1024, 3)) / sizeof(float));
    buf << "Blueprint " << bp_idx << " buffer: " << std::setprecision(2) << std::fixed << buf_sz << " elements";
    Logger::dump(buf);

    if(regrets.data().size() > meta.max_regrets) {
      meta.max_regrets = regrets.data().size();
      Logger::log("New max regrets: " + std::to_string(meta.max_regrets));
      meta.history_map = regrets.history_map();
      Logger::log("New history map size: " + std::to_string(meta.history_map.size()));
    }

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
          auto freq = calculate_strategy(regrets, base_idx, n_actions);
          for(int fidx = 0; fidx < freq.size(); ++fidx) {
            buffer.freqs[base_idx - base_idxs[bidx_start] + fidx] = freq[fidx];
          }
        }
      }
      bidx_start = bidx_end;

      Logger::log("Saving buffer " + std::to_string(buf_idx) + "...");
      std::string fn = "lossless_buf_" + std::to_string(buf_idx++) + ".bin";
      meta.buffer_fns.push_back(fn);
      cereal_save(buffer, (buffer_dir / fn).string());
      Logger::log("Saved buffer " + std::to_string(buf_idx) + " successfully.");
    }
  }
  return meta;
}

void LosslessBlueprint::build(const std::string& preflop_fn, const std::vector<std::string>& postflop_fns, const std::string& buf_dir) {
  Logger::log("Building lossless blueprint...");
  std::filesystem::path buffer_dir = buf_dir;
  LosslessMetadata meta = build_lossless_buffers(preflop_fn, postflop_fns, buf_dir);
  set_config(meta.config);
  assign_freq(new StrategyStorage<float>{meta.config.action_profile, meta.n_clusters});
  get_freq()->data().resize(meta.max_regrets);
  for(std::string buf_fn : meta.buffer_fns) {
    LosslessBuffer buf;
    cereal_load(buf, (buffer_dir / buf_fn).string());
    Logger::log("Accumulating " + buf_fn + ": [" + std::to_string(buf.offset) + ", " + std::to_string(buf.offset + buf.freqs.size()) + ")");
    #pragma omp parallel for schedule(static)
    for(size_t idx = 0; idx < buf.freqs.size(); ++idx) {
      auto& entry = get_freq()->operator[](buf.offset + idx);
      auto regret = entry.load();
      if(regret >= 1'000'000'000) Logger::error("Lossless blueprint regret accumulation overflow! Regret=" + std::to_string(regret));
      entry.store(regret + buf.freqs[idx]);
    }
  }

  Logger::log("Inserting histories...");
  for(const auto& entry : meta.history_map) {
    get_freq()->history_map()[entry.first] = entry.second;
  }

  Logger::log("Normalizing frequencies...");
  std::vector<size_t> base_idxs = _collect_base_indexes(*get_freq());
  for(size_t curr_idx = 0; curr_idx < base_idxs.size() - 1; ++curr_idx) {
    if(curr_idx + 1 >= base_idxs.size()) Logger::error("Renorm: Indexing base indeces out of range!");
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
          get_freq()->operator[](base_idx + aidx).store(get_freq()->operator[](base_idx + aidx).load() / total);
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

struct SampledMetadata {
  BlueprintTrainerConfig config;
  std::vector<std::string> buffer_fns;
  std::vector<ActionHistory> histories;
  std::vector<Action> biases;
  int n_clusters = -1;
};

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

SampledMetadata build_sampled_buffers(const std::string& lossless_bp_fn, const std::string& buf_dir, float factor) {
  std::cout << "Building sampled buffers...\n";
  SampledMetadata meta;
  std::filesystem::path buffer_dir = buf_dir;
  LosslessBlueprint bp;
  cereal_load(bp, lossless_bp_fn);
  meta.config = bp.get_config();
  BiasActionProfile bias_profile;
  meta.biases = bias_profile.get_actions(0, 0, 0, 150);
  meta.n_clusters = bp.get_strategy().n_clusters();

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
  std::unordered_map<ActionHistory, std::vector<Action>> buffer;
  long long min_ram = 4 * pow(1024, 3);
  int buf_idx = 0;
  for(size_t hidx = 0; hidx < meta.histories.size(); ++hidx) {
    std::vector<Action> sampled;
    sampled.reserve(meta.n_clusters * meta.biases.size());
    for(int c = 0; c < meta.n_clusters; ++c) {
      PokerState state = init_state.apply(meta.histories[hidx]);
      auto actions = valid_actions(state, bp.get_config().action_profile);
      size_t base_idx = bp.get_strategy().index(state, c);
      auto freq = calculate_strategy(bp.get_strategy(), base_idx, actions.size());
      for(Action bias : meta.biases) {
        sampled.push_back(sample_biased(actions, freq, bias, factor));
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

void SampledBlueprint::build(const std::string& lossless_bp_fn, const std::string& buf_dir, float bias_factor) {
  SampledMetadata meta = build_sampled_buffers(lossless_bp_fn, buf_dir, bias_factor);
  std::filesystem::path buffer_dir = buf_dir;
  assign_freq(new StrategyStorage<Action>(BiasActionProfile{}, meta.n_clusters));
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
}

}
