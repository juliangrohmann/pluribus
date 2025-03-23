#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <sys/sysinfo.h>
#include <pluribus/cereal_ext.hpp>
#include <pluribus/util.hpp>
#include <pluribus/mccfr.hpp>
#include <pluribus/range_viewer.hpp>
#include <pluribus/blueprint.hpp>

namespace pluribus {

void Blueprint::build(const std::string& preflop_fn, const std::vector<std::string>& postflop_fns, const std::string& buf_dir) {
  std::filesystem::path buffer_dir = buf_dir;
  size_t max_regrets = 0;
  tbb::concurrent_unordered_map<ActionHistory, HistoryEntry> history_map;
  int n_clusters;

  std::vector<std::string> buffer_fns;
  int buf_idx = 0;

  for(int bp_idx = 0; bp_idx < postflop_fns.size(); ++bp_idx) {
    auto bp = cereal_load<BlueprintTrainer>(postflop_fns[bp_idx]);
    const auto& regrets = bp.get_strategy();
    if(bp_idx == 0) {
      _config = bp.get_config();
      n_clusters = regrets.n_clusters();
      std::cout << "Initialized blueprint config:\n";
      std::cout << "n_clusters=" << n_clusters << "\n";
      std::cout << "max_actions=" << _config.action_profile.max_actions() << "\n";
      std::cout << _config.to_string();
    }

    std::vector<size_t> base_idxs;
    base_idxs.reserve(regrets.history_map().size());
    for(const auto& entry : regrets.history_map()) {
      base_idxs.push_back(entry.second.idx);
    }
    base_idxs.push_back(regrets.data().size());
    std::sort(base_idxs.begin(), base_idxs.end());

    size_t free_ram = get_free_ram();
    if(free_ram < 8192) {
      throw std::runtime_error("At least 8G free RAM required to build blueprint. Available (bytes): " + std::to_string(free_ram));
    }
    size_t buf_sz = static_cast<size_t>((free_ram - 4096) / sizeof(float));
    std::cout << "Blueprint " << bp_idx << " buffer: " << std::setprecision(2) << std::fixed << buf_sz << " elements\n";

    if(regrets.data().size() > max_regrets) {
      max_regrets = regrets.data().size();
      std::cout << "New max regrets: " << max_regrets << "\n";
      history_map = regrets.history_map();
    }

    size_t bidx_start = 0;
    size_t bidx_end = 0;
    while(bidx_start < base_idxs.size() - 1) {
      for(; bidx_end < base_idxs.size() - 1; ++bidx_end) {
        if(base_idxs[bidx_end] - base_idxs[bidx_start] > buf_sz) break;
      }
      if(bidx_end == bidx_start) throw std::runtime_error("Failed to increase bidx_end");

      Buffer buffer;
      buffer.offset = base_idxs[bidx_start];
      buffer.freqs.resize(base_idxs[bidx_end] - base_idxs[bidx_start]);

      std::cout << "Storing buffer: base_idxs[" << bidx_start << ", " << bidx_end << "), indeces: [" 
                << base_idxs[bidx_start] << ", " << base_idxs[bidx_end] << ")\n";
      #pragma omp parallel for schedule(dynamic)
      for(size_t curr_idx = bidx_start; curr_idx < bidx_end; ++curr_idx) {
        if(curr_idx + 1 >= base_idxs.size()) throw std::runtime_error("Buffering: Indexing base indeces out of range!");
        size_t n_entries = base_idxs[curr_idx + 1] - base_idxs[curr_idx];
        if(n_entries % n_clusters != 0) throw std::runtime_error("Buffering: Indivisible regret section!");
        size_t n_actions = n_entries / n_clusters;
        if(n_actions > _config.action_profile.max_actions()) throw std::runtime_error("Buffering: Too many actions in storage section:" + 
            std::to_string(n_actions) + " > " + std::to_string(_config.action_profile.max_actions()));
        for(int c = 0; c < n_clusters; ++c) {
          size_t base_idx = base_idxs[curr_idx] + c * n_actions;
          auto freq = calculate_strategy(regrets, base_idx, n_actions);
          for(int fidx = 0; fidx < freq.size(); ++fidx) {
            buffer.freqs[base_idx - base_idxs[bidx_start] + fidx] = freq[fidx];
          }
        }
      }
      bidx_start = bidx_end;

      std::string fn = "buf_" + std::to_string(buf_idx++) + ".bin";
      buffer_fns.push_back(fn);
      cereal_save(buffer, (buffer_dir / fn).string());
    }
  }

  _freq = std::unique_ptr<StrategyStorage<float>>{new StrategyStorage<float>{_config.action_profile, n_clusters}};
  _freq->data().resize(max_regrets);
  for(std::string buf_fn : buffer_fns) {
    auto buf = cereal_load<Buffer>((buffer_dir / buf_fn).string());
    std::cout << "Accumulating " << buf_fn << ": [" << buf.offset << ", " << buf.offset + buf.freqs.size() << ")\n";
    #pragma omp parallel for schedule(static)
    for(size_t idx = 0; idx < buf.freqs.size(); ++idx) {
      _freq->operator[](buf.offset + idx).store(_freq->operator[](buf.offset + idx).load() + buf.freqs[idx]);
    }
  }

  std::cout << "Inserting histories...\n";
  for(const auto& entry : history_map) {
    _freq->history_map()[entry.first] = entry.second;
  }

  std::cout << "Normalizing frequencies...\n";
  std::vector<size_t> base_idxs;
  base_idxs.reserve(history_map.size());
  for(const auto& entry : history_map) {
    base_idxs.push_back(entry.second.idx);
  }
  base_idxs.push_back(_freq->data().size());
  std::sort(base_idxs.begin(), base_idxs.end());

  for(size_t curr_idx = 0; curr_idx < base_idxs.size() - 1; ++curr_idx) {
    if(curr_idx + 1 >= base_idxs.size()) throw std::runtime_error("Renorm: Indexing base indeces out of range!");
    size_t n_entries = base_idxs[curr_idx + 1] - base_idxs[curr_idx];
    if(n_entries % n_clusters != 0) throw std::runtime_error("Renorm: Indivisible storage section!");
    size_t n_actions = n_entries / n_clusters;
    if(n_actions > _config.action_profile.max_actions()) throw std::runtime_error("Renorm: Too many actions in storage section!");
    for(int c = 0; c < n_clusters; ++c) {
      size_t base_idx = base_idxs[curr_idx] + c * n_actions;
      float total = 0.0f;
      for(size_t aidx = 0; aidx < n_actions; ++aidx) {
        total += _freq->operator[](base_idx + aidx);
      }
      for(size_t aidx = 0; aidx < n_actions; ++aidx) {
        _freq->operator[](base_idx + aidx) / total;
      }
    }
  }
}

}
