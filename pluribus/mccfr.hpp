#pragma once

#include <array>
#include <vector>
#include <atomic>
#include <memory>
#include <filesystem>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cereal/cereal.hpp>
#include <libwandb_cpp.h>
#include <tbb/concurrent_vector.h>
#include <tbb/concurrent_unordered_map.h>
#include <pluribus/range.hpp>
#include <pluribus/cereal_ext.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/infoset.hpp>
#include <pluribus/storage.hpp>


namespace pluribus {

// using PreflopMap = tbb::concurrent_unordered_map<InformationSet, tbb::concurrent_vector<float>>;

template <class T>
std::vector<float> calculate_strategy(const StrategyStorage<T>& data, size_t base_idx, int n_actions) {
  T sum = 0;
  for(int a_idx = 0; a_idx < n_actions; ++a_idx) {
    sum += std::max(data[base_idx + a_idx].load(), static_cast<T>(0));
  }

  std::vector<float> freq;
  freq.reserve(n_actions);
  if(sum > 0) {
    for(int a_idx = 0; a_idx < n_actions; ++a_idx) {
      freq.push_back(std::max(data[base_idx + a_idx].load(), static_cast<T>(0)) / static_cast<double>(sum));
    }
  }
  else {
    for(int i = 0; i < n_actions; ++i) {
      freq.push_back(1.0 / n_actions);
    }
  }
  return freq;
}

template <class T>
void lcfr_discount(StrategyStorage<T>& regrets, double d) {
  for(auto& e : regrets.data()) {
    e.store(e.load() * d);
  }
}

int sample_action_idx(const std::vector<float>& freq);

struct BlueprintTimingConfig {
  long preflop_threshold_m = 800;
  long snapshot_interval_m = 200;
  long prune_thresh_m = 200;
  long lcfr_thresh_m = 400;
  long discount_interval_m = 10;
  long log_interval_m = 1;
};

struct BlueprintTrainerConfig {
  BlueprintTrainerConfig(int n_players = 2, int n_chips = 10'000, int ante = 0);
  BlueprintTrainerConfig(const PokerConfig& poker_);

  std::string to_string() const;

  bool operator==(const BlueprintTrainerConfig&) const = default;

  void set_iterations(const BlueprintTimingConfig& timings, long it_per_min) {
    preflop_threshold = timings.preflop_threshold_m * it_per_min;
    snapshot_interval = timings.snapshot_interval_m * it_per_min;
    prune_thresh = timings.prune_thresh_m * it_per_min;
    lcfr_thresh = timings.lcfr_thresh_m * it_per_min;
    discount_interval = timings.discount_interval_m * it_per_min;
    log_interval = timings.log_interval_m * it_per_min;
  }

  template <class Archive>
  void serialize(Archive& ar) {
    ar(poker, action_profile, init_ranges, init_board, init_state, strategy_interval, preflop_threshold, snapshot_interval, 
       prune_thresh, lcfr_thresh, discount_interval, log_interval, prune_cutoff, regret_floor);
  }

  PokerConfig poker;
  ActionProfile action_profile = BlueprintActionProfile{};
  std::vector<PokerRange> init_ranges;
  std::vector<uint8_t> init_board;
  PokerState init_state;
  long strategy_interval = 10'000;
  long preflop_threshold;
  long snapshot_interval;
  long prune_thresh;
  long lcfr_thresh;
  long discount_interval;
  long log_interval;
  int prune_cutoff = -300'000'000;
  int regret_floor = -310'000'000;
};

class BlueprintTrainer {
public:
  BlueprintTrainer(const BlueprintTrainerConfig& config = BlueprintTrainerConfig{}, const std::string& snapshot_dir = "", bool enable_wandb = false);
  void mccfr_p(long T);
  bool operator==(const BlueprintTrainer& other) const;
  const StrategyStorage<int>& get_regrets() const { return _regrets; }
  StrategyStorage<int>& get_regrets() { return _regrets; }
  const StrategyStorage<float>& get_phi() const { return _phi; }
  const BlueprintTrainerConfig& get_config() const { return _config; }
  void set_snapshot_dir(std::string snapshot_dir) { _snapshot_dir = snapshot_dir; }
  void set_verbose(bool verbose) { _verbose = verbose; }
  void set_verbose_update(bool verbose_update) { _verbose_update = verbose_update; }

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_regrets, _phi, _config, _t);
  }

private:
  int traverse_mccfr_p(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, const omp::HandEvaluator& eval);
  int traverse_mccfr(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, const omp::HandEvaluator& eval);
  void update_strategy(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, PokerRange& update_range);
  int utility(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, const omp::HandEvaluator& eval) const;
  int showdown_payoff(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, const omp::HandEvaluator& eval) const;
  void log_metrics(long t);

#ifdef UNIT_TEST
  friend int call_traverse_mccfr(BlueprintTrainer& trainer, const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, 
                                 const omp::HandEvaluator& eval);
  friend void call_update_strategy(BlueprintTrainer& trainer, const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands);
#endif
  StrategyStorage<int> _regrets;
  StrategyStorage<float> _phi;
  BlueprintTrainerConfig _config;
  std::filesystem::path _snapshot_dir;
  std::unique_ptr<wandb::Session> _wb;
  wandb::Run _wb_run;
  long _t;
  bool _verbose = false;
  bool _verbose_update = false;
};

}
