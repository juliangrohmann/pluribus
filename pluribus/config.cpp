#include <pluribus/config.hpp>
#include <pluribus/debug.hpp>

namespace pluribus {

SolverConfig::SolverConfig(const PokerConfig& poker_, const ActionProfile& action_profile)
    : poker{poker_}, rake{0.0, 0.0}, action_profile{action_profile}, init_state{poker_}, restrict_players{poker_.n_players} {
  for(int i = 0; i < poker_.n_players; ++i) init_ranges.push_back(PokerRange::full());
}

std::string ranges_to_config_str(const std::vector<PokerRange>& ranges) {
  std::ostringstream oss;
  for(int i = 0; i < ranges.size(); ++ i) oss << "Player " << i << ": " << ranges[i].n_combos() << " combos\n";
  return oss.str();
}

std::string SolverConfig::to_string() const {
  std::ostringstream oss;
  oss << "================ MCCFR Config ================\n"
      << "Poker config: " << poker.to_string() << "\n"
      << "Initial board: " << cards_to_str(init_board.data(), init_board.size()) << "\n"
      << "Initial state:\n" << init_state.to_string() << "\n"
      << "Initial ranges:\n"
      << ranges_to_config_str(init_ranges) << "\n"
      << "Dead ranges:\n"
      << ranges_to_config_str(dead_ranges) << "\n"
      << "Action profile:\n" << action_profile.to_string()
      << "----------------------------------------------------------\n";
  return oss.str();
}

long DiscountConfig::next_discount_step(const long t, const long T) const {
  const long next_disc = (t / discount_interval + 1) * discount_interval;
  return next_disc < lcfr_thresh ? next_disc : T + 1;
}

bool DiscountConfig::is_discount_step(const long t) const {
  return t < lcfr_thresh && t % discount_interval == 0;
}

double DiscountConfig::get_discount_factor(const long t) const {
  return static_cast<double>(t / discount_interval) / (t / discount_interval + 1);
}

BlueprintSolverConfig::BlueprintSolverConfig(const BlueprintTimingConfig& timings, const long it_per_min) {
  set_iterations(timings, it_per_min);
}

std::string BlueprintSolverConfig::to_string() const {
  std::ostringstream oss;
  oss << "================ Blueprint Trainer Config ================\n";
  oss << "Strategy interval: " << strategy_interval << "\n";
  oss << "Preflop threshold: " << preflop_threshold << "\n";
  oss << "Snapshot threshold: " << snapshot_threshold << "\n";
  oss << "Snapshot interval: " << snapshot_interval << "\n";
  oss << "Prune threshold: " << prune_thresh << "\n";
  oss << "LCFR threshold: " << lcfr_thresh << "\n";
  oss << "Discount interval: " << discount_interval << "\n";
  oss << "Log interval: " << log_interval << "\n";
  oss << "----------------------------------------------------------\n";
  return oss.str();
}

void BlueprintSolverConfig::set_iterations(const BlueprintTimingConfig& timings, const long it_per_min) {
  preflop_threshold = timings.preflop_thresh_m * it_per_min;
  snapshot_threshold = timings.snapshot_thresh_m * it_per_min;
  snapshot_interval = timings.snapshot_interval_m * it_per_min;
  prune_thresh = timings.prune_thresh_m * it_per_min;
  lcfr_thresh = timings.lcfr_thresh_m * it_per_min;
  discount_interval = timings.discount_interval_m * it_per_min;
  log_interval = timings.log_interval_m * it_per_min;
}

long BlueprintSolverConfig::next_snapshot_step(const long t, const long T) const {
    if(t < snapshot_threshold) return snapshot_threshold;
    const long next_snap = std::max((t - snapshot_threshold) / snapshot_interval + 1, 0L) * snapshot_interval + snapshot_threshold;
    return next_snap < T ? next_snap : T;
}

bool BlueprintSolverConfig::is_snapshot_step(const long t, const long T) const {
  return t == T || (t >= snapshot_threshold && (t - snapshot_threshold) % snapshot_interval == 0);
}

RealTimeSolverConfig::RealTimeSolverConfig(const RealTimeTimingConfig& timings, const long it_per_sec) {
  set_iterations(timings, it_per_sec);
}

std::string RealTimeSolverConfig::to_string() const {
  return "Terminal round: " + round_to_str(terminal_round) + ", Terminal bet level: " + std::to_string(terminal_bet_level) + "-bet";
}

void RealTimeSolverConfig::set_iterations(const RealTimeTimingConfig& timings, const long it_per_sec) {
  discount_interval = timings.discount_interval_s * it_per_sec;
  lcfr_thresh = timings.lcfr_thresh_s * it_per_sec;
  log_interval = timings.log_interval_s * it_per_sec;
}

}
