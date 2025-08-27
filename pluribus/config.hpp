#pragma once

#include <pluribus/actions.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/profiles.hpp>
#include <pluribus/range.hpp>

namespace pluribus {

struct SolverConfig {
  explicit SolverConfig(const PokerConfig& poker_ = PokerConfig{}, const ActionProfile& action_profile = ActionProfile{},
    const std::vector<int>& stacks = std::vector<int>{});
  explicit SolverConfig(const PokerConfig& poker_, const ActionProfile& action_profile, int stacks);

  int infer_stack_size(const int i) const { return init_state.get_players()[i].get_chips() + init_state.get_players()[i].get_betsize(); }
  std::string to_string() const;

  bool operator==(const SolverConfig& other) const = default;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(poker, rake, action_profile, init_ranges, dead_ranges, init_board, init_chips, init_state, restrict_players);
  }

  PokerConfig poker;
  RakeStructure rake;
  ActionProfile action_profile;
  std::vector<PokerRange> init_ranges;
  std::vector<PokerRange> dead_ranges;
  std::vector<uint8_t> init_board;
  std::vector<int> init_chips;
  PokerState init_state;
  int restrict_players;
};

class ConfigProvider {
public:
  virtual const SolverConfig& get_config() const = 0;
  virtual ~ConfigProvider() = default;
};

struct DiscountConfig {
  long next_discount_step(long t, long T) const;
  bool is_discount_step(long t) const;
  double get_discount_factor(long t) const;

  bool operator==(const DiscountConfig& other) const = default;

  long discount_interval;
  long lcfr_thresh;
};

struct BlueprintTimingConfig {
  long discount_interval_m = 10;
  long lcfr_thresh_m = 400;
  long preflop_thresh_m = 800;
  long snapshot_thresh_m = 800;
  long snapshot_interval_m = 200;
  long prune_thresh_m = 200;
  long log_interval_m = 1;
};

struct BlueprintSolverConfig : DiscountConfig {
  explicit BlueprintSolverConfig(const BlueprintTimingConfig& timings = BlueprintTimingConfig{}, long it_per_min = 10'000'000L);

  std::string to_string() const;

  void set_iterations(const BlueprintTimingConfig& timings, long it_per_min);

  long next_snapshot_step(long t, long T) const;
  bool is_snapshot_step(long t, long T) const;

  bool operator==(const BlueprintSolverConfig& other) const = default;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(strategy_interval, preflop_threshold, snapshot_threshold, snapshot_interval, prune_thresh, lcfr_thresh, discount_interval, log_interval);
  }

  long strategy_interval = 10'000;
  long preflop_threshold = -1;
  long snapshot_threshold = -1;
  long snapshot_interval = -1;
  long prune_thresh = -1;
  long log_interval = -1;
};

struct RealTimeTimingConfig {
  double discount_interval_s = 0.5;
  double lcfr_thresh_s = 15.0;
  double log_interval_s = 1.0;
};

struct RealTimeSolverConfig : DiscountConfig {
  explicit RealTimeSolverConfig(const RealTimeTimingConfig& timings = RealTimeTimingConfig{}, long it_per_sec = 750'000);

  std::string to_string() const;
  void set_iterations(const RealTimeTimingConfig& timings, long it_per_sec);
  bool is_terminal() const { return terminal_round >= 4 && terminal_bet_level >= 99; }

  bool operator==(const RealTimeSolverConfig& other) const = default;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(bias_profile, log_interval, terminal_round, terminal_bet_level);
  }

  ActionProfile bias_profile = BiasActionProfile{};
  std::vector<Action> init_actions;
  long log_interval = -1;
  int terminal_round = -1;
  int terminal_bet_level = -1;
};

}
