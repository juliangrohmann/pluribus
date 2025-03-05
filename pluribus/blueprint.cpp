#include <iostream>
#include <iomanip>
#include <cmath>
#include <sys/sysinfo.h>
#include <pluribus/cereal_ext.hpp>
#include <pluribus/util.hpp>
#include <pluribus/mccfr.hpp>
#include <pluribus/blueprint.hpp>

namespace pluribus {

void Blueprint::build(const std::string& preflop_fn, const std::vector<std::string>& postflop_fns, const std::string& buf_dir) {
  std::filesystem::path buffer_dir = buf_dir;
  size_t max_regrets = 0;
  tbb::concurrent_unordered_map<ActionHistory, int> history_map;
  BlueprintTrainerConfig config;
  int n_clusters;

  std::vector<std::string> buffer_fns;
  int buf_idx = 0;

  for(int bp_idx = 0; bp_idx < postflop_fns.size(); ++bp_idx) {
    auto bp = cereal_load<BlueprintTrainer>(postflop_fns[bp_idx]);
    const auto& regrets = bp.get_regrets();
    if(bp_idx = 0) {
      config = bp.get_config();
      n_clusters = regrets.n_clusters();
      std::cout << "Initialized blueprint config.";
    }

    size_t buf_sz = static_cast<size_t>(0.8 * (get_free_ram() / sizeof(std::atomic<int>)));
    std::cout << "Blueprint " << bp_idx << " buffer: " << std::setprecision(2) << std::fixed << buf_sz / static_cast<double>(pow(1024, 3)) << "\n";

    if(regrets.data().size() > max_regrets) {
      max_regrets = regrets.data().size();
      std::cout << "New max regrets: " << max_regrets << "\n";
      history_map = regrets.history_map();
    }

    size_t offset = 0;
    while(offset < regrets.data().size()) {
      size_t curr_buf_sz = std::min(regrets.data().size() - offset, buf_sz);
      Buffer buffer;
      buffer.offset = offset;
      buffer.regrets.resize(curr_buf_sz);

      std::cout << "Storing buffer " << buf_idx << ": [" << offset << ", " << offset + curr_buf_sz << ")\n";
      #pragma omp parallel for schedule(static)
      for(size_t idx = 0; idx < curr_buf_sz ; ++idx) {
        buffer.regrets[idx] = regrets.data()[offset + idx];
      }

      offset += curr_buf_sz;
      std::string fn = "buf_" + std::to_string(buf_idx++) + ".bin";
      buffer_fns.push_back(fn);
      cereal_save(buffer, (buffer_dir / fn).string());
    }
  }
  
  StrategyStorage<int> cum_strategy;
  cum_strategy.data().resize(max_regrets);
  for(std::string buf_fn : buffer_fns) {
    auto buf = cereal_load<Buffer>((buffer_dir / buf_fn).string());
    std::cout << "Accumulating " << buf_fn << ": [" << buf.offset << ", " << buf.offset + buf.regrets.size() << ")\n";
    #pragma omp parallel for schedule(static)
    for(size_t idx = 0; idx < buf.regrets.size(); ++idx) {
      cum_strategy[buf.offset + idx].store(cum_strategy[buf.offset + idx].load() + buf.regrets[idx]);
    }
  }

  std::cout << "Computing frequencies...\n";
  _freq = std::unique_ptr<StrategyStorage<float>>{new StrategyStorage<float>};
  for(const auto& entry : history_map) {
    PokerState state{config.poker};
    state = state.apply(entry.first);
    auto actions = valid_actions(state, config.action_profile);

    for(int c = 0; c < n_clusters; ++c) {
      int regret_idx = entry.second + c * actions.size();
      auto curr_freq = calculate_strategy(cum_strategy, regret_idx, actions.size());
      size_t freq_idx = _freq->index(state, c);
      for(int a_idx = 0; a_idx < actions.size(); ++a_idx) {
        _freq->operator[](freq_idx + a_idx).store(curr_freq[a_idx]);
      }
    }
  }
}

}
