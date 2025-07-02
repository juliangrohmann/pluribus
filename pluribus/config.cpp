#include <pluribus/config.hpp>
#include <pluribus/debug.hpp>

namespace pluribus {

SolverConfig::SolverConfig(const PokerConfig& poker_) 
    : poker{poker_}, action_profile{BlueprintActionProfile{poker_.n_players}}, init_state{poker_} {
  for(int i = 0; i < poker_.n_players; ++i) init_ranges.push_back(PokerRange::full());
}

SolverConfig::SolverConfig(int n_players, int n_chips, int ante) 
    : SolverConfig{PokerConfig{n_players, n_chips, ante}} {}

std::string SolverConfig::to_string() const {
  std::ostringstream oss;
  oss << "================ MCCFR Config ================\n";
  oss << "Poker config: " << poker.to_string() << "\n";
  oss << "Initial board: " << cards_to_str(init_board.data(), init_board.size()) << "\n";
  oss << "Initial state:\n" << init_state.to_string() << "\n";
  oss << "Initial ranges:\n";
  for(int i = 0; i < init_ranges.size(); ++ i) oss << "Player " << i << ": " << init_ranges[i].n_combos() << " combos\n";
  oss << "Action profile:\n" << action_profile.to_string();
  oss << "----------------------------------------------------------\n";
  return oss.str();
}

long DiscountConfig::next_discount_step(long t, long T) const {
  long next_disc = (t / discount_interval + 1) * discount_interval;
  return next_disc < lcfr_thresh ? next_disc : T + 1;
}

bool DiscountConfig::is_discount_step(long t) const {
  return t < lcfr_thresh && t % discount_interval == 0;
}

double DiscountConfig::get_discount_factor(long t) const {
  return static_cast<double>(t / discount_interval) / (t / discount_interval + 1);
}

BlueprintSolverConfig::BlueprintSolverConfig(const BlueprintTimingConfig& timings, long it_per_min) {
  set_iterations(timings, it_per_min);
}

std::string BlueprintSolverConfig::to_string() const {
  std::ostringstream oss;
  oss << "================ Blueprint Trainer Config ================\n";
  oss << "Strategy interval: " << strategy_interval << "\n";
  oss << "Preflop threshold: " << preflop_threshold << "\n";
  oss << "Snapshot interval: " << snapshot_interval << "\n";
  oss << "Prune threshold: " << prune_thresh << "\n";
  oss << "LCFR threshold: " << lcfr_thresh << "\n";
  oss << "Discount interval: " << discount_interval << "\n";
  oss << "Log interval: " << log_interval << "\n";
  oss << "----------------------------------------------------------\n";
  return oss.str();
}

void BlueprintSolverConfig::set_iterations(const BlueprintTimingConfig& timings, long it_per_min) {
  preflop_threshold = timings.preflop_threshold_m * it_per_min;
  snapshot_interval = timings.snapshot_interval_m * it_per_min;
  prune_thresh = timings.prune_thresh_m * it_per_min;
  lcfr_thresh = timings.lcfr_thresh_m * it_per_min;
  discount_interval = timings.discount_interval_m * it_per_min;
  log_interval = timings.log_interval_m * it_per_min;
}

long BlueprintSolverConfig::next_snapshot_step(long t, long T) const {
    if(t < preflop_threshold) return preflop_threshold;
    long next_snap = std::max((t - preflop_threshold) / snapshot_interval + 1, 0L) * snapshot_interval + preflop_threshold;
    return next_snap < T ? next_snap : T;
}

bool BlueprintSolverConfig::is_snapshot_step(long t, long T) const {
  return t == T || (t - preflop_threshold) % snapshot_interval == 0;
}

RealTimeSolverConfig::RealTimeSolverConfig(const RealTimeTimingConfig& timings, long it_per_sec) {
  set_iterations(timings, it_per_sec);
}

std::string RealTimeSolverConfig::to_string() const {
  return "Terminal round: " + round_to_str(terminal_round) + ", Terminal bet level: " + std::to_string(terminal_bet_level) + "-bet";
}

void RealTimeSolverConfig::set_iterations(const RealTimeTimingConfig& timings, long it_per_sec) {
  discount_interval = timings.discount_interval_s * it_per_sec;
  lcfr_thresh = timings.lcfr_thresh_s * it_per_sec;
  log_interval = timings.log_interval_s * it_per_sec;
}

}
