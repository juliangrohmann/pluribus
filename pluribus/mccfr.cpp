#include <algorithm>
#include <atomic>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <omp.h>
#include <string>
#include <json/json.hpp>
#include <pluribus/actions.hpp>
#include <pluribus/cluster.hpp>
#include <pluribus/debug.hpp>
#include <pluribus/decision.hpp>
#include <pluribus/logging.hpp>
#include <pluribus/mccfr.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/rng.hpp>
#include <pluribus/traverse.hpp>
#include <pluribus/util.hpp>
#include <tqdm/tqdm.hpp>

#include "translate.hpp"

namespace pluribus {

static constexpr int PRUNE_CUTOFF = -300'000'000;
static constexpr int REGRET_FLOOR = -310'000'000;
static constexpr int MAX_ACTIONS = 16;

int utility(const SlimPokerState& state, const int i, const Board& board, const std::vector<Hand>& hands, const int stack_size, const RakeStructure& rake,
    const omp::HandEvaluator& eval) {
  if(state.get_players()[i].has_folded()) {
    return state.get_players()[i].get_chips() - stack_size;
  }
  if(state.get_winner() != -1) { // TODO: rake is inflated when a player places an unnecessarily large all-in, take back chips instead of counting as pot
    return state.get_players()[i].get_chips() - stack_size + (state.get_winner() == i ? rake.payoff(state.get_round(), state.get_pot().total()) : 0);
  }
  if(state.get_round() >= 4) {
    return state.get_players()[i].get_chips() - stack_size + showdown_payoff(state, i, board, hands, rake, eval);
  }
  Logger::error("Non-terminal state does not have utility.");
}

Solver::Solver(const SolverConfig& config) : _config{config} {
  if(config.init_board.size() != n_board_cards(config.init_state.get_round())) {
    Logger::error("Wrong amount of solver board cards. Round=" + round_to_str(config.init_state.get_round()) + 
        ", Board=" + cards_to_str(config.init_board));
  }
  if(config.init_state.get_players().size() != config.poker.n_players) Logger::error("Player number mismatch in Solver.");
}

void Solver::solve(const long t_plus) {
  Logger::log("================================= Solve ==================================");
  _state = SolverState::SOLVING;
  _solve(t_plus);
  _state = SolverState::SOLVED;
}

// ==========================================================================================
// || MCCFRSolver
// ==========================================================================================

template <class T>
std::vector<float> state_to_freq(std::atomic<T>* base_ptr, int n_actions) {
  return calculate_strategy(base_ptr, n_actions);
}

template <template<typename> class StorageT>
void MCCFRSolver<StorageT>::_solve(long t_plus) {
  if(!create_dir(_snapshot_dir)) Logger::error("Failed to create snapshot dir: " + _snapshot_dir.string());
  if(!create_dir(_metrics_dir)) Logger::error("Failed to create metrics dir: " + _metrics_dir.string());
  if(!create_dir(_log_dir)) Logger::error("Failed to create log dir: " + _log_dir.string());

  const int max_actions = get_config().action_profile.max_actions();
  if(max_actions > MAX_ACTIONS) Logger::error("Action profile max actions is too large: " + std::to_string(max_actions) + " > " + std::to_string(MAX_ACTIONS));

  long T = _t + t_plus;
  Logger::log((HoleCardIndexer::get_instance() ? "Initialized" : "Failed to initialize") + std::string{" hole card indexer."});
  Logger::log((HandIndexer::get_instance() ? "Initialized" : "Failed to initialize") + std::string{" hand indexer."});
  on_start();

  Logger::log("Training blueprint from " + std::to_string(_t) + " to " + std::to_string(T));
  std::ostringstream buf;
  while(_t < T) {
    long init_t = _t;
    _t = next_step(_t, T); 
    auto interval_start = std::chrono::high_resolution_clock::now();
    buf << std::setprecision(1) << std::fixed << "Next step: " << _t / 1'000'000.0 << "M";
    Logger::dump(buf);
    auto t_0 = std::chrono::high_resolution_clock::now();
    if(is_debug) omp_set_num_threads(1);
    #pragma omp parallel for schedule(dynamic, 1)
    for(long t = init_t; t < _t; ++t) {
      if(is_interrupted()) continue;
      thread_local omp::HandEvaluator eval;
      thread_local Board board;
      thread_local MarginalRejectionSampler sampler{get_config().init_ranges, get_config().init_board, get_config().dead_ranges};
      if(is_debug) Logger::log("============== t = " + std::to_string(t) + " ==============");
      if(should_log(t)) {
        std::ostringstream metrics_fn;
        metrics_fn << std::setprecision(1) << std::fixed << t / 1'000'000.0 << ".json";
        write_to_file(_metrics_dir / metrics_fn.str(), track_wandb_metrics(t));
        Logger::log(progress_str(t - init_t, _t - init_t, t_0));
      }
      for(int i = 0; i < get_config().poker.n_players; ++i) {
        if(is_debug) Logger::log("============== i = " + std::to_string(i) + " ==============");
        std::vector<CachedIndexer> indexers(get_config().poker.n_players);
        RoundSample sample = sampler.sample();
        board = sample_board(get_config().init_board, sample.mask);
        for(int h_idx = 0; h_idx < sample.hands.size(); ++h_idx) {
          indexers[h_idx].index(board, sample.hands[h_idx], 3); // cache indexes
        }
        on_step(t, i, sample.hands, indexers);
        SlimPokerState state{get_config().init_state};
        SlimPokerState bp_state{get_config().init_state};
        MCCFRContext<StorageT> ctx{state, t, i, 0, board, sample.hands, indexers, eval, init_regret_storage(), init_bp_node(), bp_state};
        if(should_prune(t)) {
          if(is_debug) Logger::log("============== Traverse MCCFR-P ==============");
          traverse_mccfr_p(ctx);
        }
        else {
          if(is_debug) Logger::log("============== Traverse MCCFR ==============");
          traverse_mccfr(ctx);
        }
      }
    }
    if(is_interrupted()) break;
    auto interval_end = std::chrono::high_resolution_clock::now();
    buf << "Step duration: " << std::chrono::duration_cast<std::chrono::seconds>(interval_end - interval_start).count() << " s.";
    Logger::dump(buf);
    if(should_discount(_t) && !is_interrupted()) {
      Logger::log("============== Discounting ==============");
      double d = get_discount_factor(_t);
      buf << std::setprecision(2) << std::fixed << "Discount factor: " << d;
      Logger::dump(buf);
      init_regret_storage()->lcfr_discount(d);
      if(StorageT<float>* init_avg = init_avg_storage()) init_avg->lcfr_discount(d);
    }
    if(should_snapshot(_t, T)) {
      std::ostringstream fn_stream;
      Logger::log("============== Saving snapshot ==============");
      fn_stream << date_time_str() << "_t" << std::setprecision(1) << std::fixed << _t / 1'000'000.0 << "M.bin";
      save_snapshot((_snapshot_dir / fn_stream.str()).string());
      on_snapshot();
    }
  }
  Logger::log(is_interrupted() ? "====================== Interrupted ======================" : "============== Blueprint training complete ==============");
}

template<template <typename> class StorageT>
int MCCFRSolver<StorageT>::terminal_utility(const MCCFRContext<StorageT>& context) const {
  return utility(context.state, context.i, context.board, context.hands, get_config().init_chips[context.i], get_config().rake, context.eval);
}

template <template<typename> class StorageT>
std::string info_str(const int prev_r, const int d_r, const MCCFRContext<StorageT>& context) {
  std::string str = "r=" + std::to_string(prev_r) + " + " + std::to_string(d_r) + "\nt=" + 
         std::to_string(context.t) + "\nBoard=" + context.board.to_string() + "\nHands=";
  for(const auto& hand : context.hands) str += hand.to_string() + "  ";
  str += "\n";
  return str;
}

// std::string relative_history_str(const PokerState& state, const PokerState& init_state) {
//   return state.get_action_history().slice(init_state.get_action_history().size()).to_string();
// }

template <template<typename> class StorageT>
void log_utility(const int utility, const MCCFRContext<StorageT>& ctx) {
  // context.debug << "Terminal: " << relative_history_str(context.state, init_state) << "\n";
  std::ostringstream debug;
  debug << "Terminal: Hands=[";
  for(int p_idx = 0; p_idx < ctx.state.get_players().size(); ++p_idx) {
    debug << ctx.hands[p_idx].to_string() << (p_idx != ctx.state.get_players().size() - 1 ? " " : "");
  }
  debug << "]  u(z) = " << utility;
  Logger::dump(debug);
}

void log_action_ev(const Action a, const float freq, const int ev) {
  std::ostringstream debug;
  debug << "Action EV: u(" << a.to_string() << ") @ " << std::setprecision(2) << std::fixed << freq << " = " << ev;
  Logger::dump(debug);
}

void log_net_ev(const int ev, const float ev_exact) {
  std::ostringstream debug;
  debug << "Net EV: u(sigma) = " << std::setprecision(2) << std::fixed << ev << " (exact=" << ev_exact << ")";
  Logger::dump(debug);
}

void log_regret(const Action a, const int d_r, const int next_r) {
  std::ostringstream debug;
  debug << "\tR(" << a.to_string() << ") = " << d_r << "\n";
  Logger::dump(debug);
  debug << "\tcum R(" << a.to_string() << ") = " << next_r;
  Logger::dump(debug);
}

void log_external_sampling(const Action sampled, const std::vector<Action>& actions, const float freq[]) {
  std::ostringstream debug;
  debug << "Sampling: ";
  for(int a_idx = 0; a_idx < actions.size(); ++a_idx) {
    debug << std::setprecision(2) << std::fixed << actions[a_idx].to_string() << "=" << freq[a_idx] << "  ";
  }
  Logger::dump(debug);
  Logger::log("Sampled: " + sampled.to_string());
}

bool should_restrict(const std::vector<Action>& actions, const int restrict_players) {
  if(actions.size() < restrict_players) return false;
  for(int i = 0; i < restrict_players; ++i) {
    if(actions[i] != Action::FOLD) return false;
  }
  return true;
}

inline bool is_terminal_call(const Action a, const int i, const SlimPokerState& state) {
  if(a != Action::CHECK_CALL) return false;
  for(int p_idx = 0; p_idx < state.get_players().size(); ++p_idx) {
    if(p_idx != i && !state.get_players()[p_idx].has_folded() && state.get_players()[p_idx].get_chips() > 0) {
      return false;
    }
  }
  return true;
}

template<class Context>
int context_cluster(const Context& ctx) {
  const int r = ctx.state.get_round();
  const int p = ctx.state.get_active();
  return BlueprintClusterMap::get_instance()->cluster(r, ctx.indexers[p].index(ctx.board, ctx.hands[p], r)); // TODO: use get_cluster
}

inline int next_consec_folds(const int consec_folds, const Action a) {
  return consec_folds > -1 && a == Action::FOLD ? consec_folds + 1 : -1;
}

template<class T>
int sample_idx_from_regrets(const std::atomic<T>* base_ptr, const int n_actions) {
  float w_local[MAX_ACTIONS];
  float S = 0.0f;
  for(int i = 0; i < n_actions; ++i) {
    const float v = static_cast<float>(base_ptr[i].load(std::memory_order_relaxed));
    const float w = v > 0.0f ? v : 0.0f;
    w_local[i] = w;
    S += w;
  }
  const float u01 = GSLGlobalRNG::uniform();
  if(S <= 0.0f) {
    const int k = static_cast<int>(u01 * n_actions);
    return k < n_actions ? k : n_actions - 1;
  }
  const float threshold = u01 * S;
  float c = 0.0f;
  for(int i=0; i<n_actions; ++i) {
    c += w_local[i];
    if(c >= threshold) return i;
  }
  return n_actions - 1;
}

template <template<typename> class StorageT>
int MCCFRSolver<StorageT>::traverse_mccfr_p(const MCCFRContext<StorageT>& ctx) {
  if(is_terminal(ctx.state, ctx.i)) {
    const int u = terminal_utility(ctx);
    if(is_debug) log_utility(u, ctx);
    return u;
  }
  if(ctx.consec_folds >= get_config().restrict_players) {
    return 0;
  }
  if(ctx.state.get_active() == ctx.i) {
    const auto& value_actions = regret_value_actions(ctx.regret_storage);
    const auto& branching_actions = regret_branching_actions(ctx.regret_storage);
    const int n_value_actions = value_actions.size();
    const int cluster = context_cluster(ctx);
    if(is_debug) Logger::log("Cluster: " + std::to_string(cluster));
    std::atomic<int>* base_ptr = get_base_regret_ptr(ctx.regret_storage, cluster);

    int values[MAX_ACTIONS];
    bool filter[MAX_ACTIONS];
    double v_exact = 0.0;
    double v_r_sum = 0.0;
    double v_a_sum = 0.0;
    int filter_sum = 0;
    for(int a_idx = 0; a_idx < n_value_actions; ++a_idx) {
      Action a = value_actions[a_idx];
      const int regret = base_ptr[a_idx].load(std::memory_order_relaxed);
      if(ctx.state.get_round() == 3 || a == Action::FOLD || regret > PRUNE_CUTOFF || is_terminal_call(a, ctx.i, ctx.state)) {
        if(is_debug) Logger::log("[" + pos_to_str(ctx.state) + "] Applying (traverser): " + a.to_string());
        filter[a_idx] = true;
        ++filter_sum;
        SlimPokerState next_state = ctx.state.apply_copy(a);
        const int branching_idx = n_value_actions == branching_actions.size() ? a_idx : 0;
        const int v_a = traverse_mccfr_p(MCCFRContext<StorageT>{next_state, next_regret_storage(ctx.regret_storage, branching_idx, next_state, ctx.i),
            next_bp_node(a, ctx.state, ctx.bp_node, ctx.bp_state), next_consec_folds(ctx.consec_folds, a), ctx});
        const int v_r = std::max(regret, 0);
        values[a_idx] = v_a;
        v_exact += static_cast<double>(v_r) * static_cast<double>(v_a);
        v_r_sum += v_r;
        v_a_sum += v_a;
        // if(is_debug) log_action_ev(a, freq[a_idx], v_a);
      }
      else {
        filter[a_idx] = false;
      }
    }
    v_exact = v_r_sum > 0 ? v_exact / v_r_sum : v_a_sum / filter_sum;
    const int v = static_cast<int>(std::lrint(v_exact));
    if(is_debug) log_net_ev(v, v_exact);
    if(!is_frozen(cluster, ctx.regret_storage)) {
      for(int a_idx = 0; a_idx < n_value_actions; ++a_idx) {
        if(filter[a_idx]) {
          auto& r_atom = base_ptr[a_idx];
          const int prev_r = r_atom.load(std::memory_order_relaxed);
          int d_r = values[a_idx] - v;
          const int next_r = prev_r + d_r;
          if(is_debug && next_r > 2'000'000'000) Logger::error("Regret overflowing!\n" + info_str(prev_r, d_r, ctx));
          if(next_r > REGRET_FLOOR) {
            r_atom.fetch_add(d_r, std::memory_order_relaxed);
          }
          if(is_debug) log_regret(value_actions[a_idx], d_r, next_r);
        }
      }
    }
    return v;
  }
  const auto& value_actions = regret_value_actions(ctx.regret_storage);
  const auto& branching_actions = regret_branching_actions(ctx.regret_storage);
  const int a_idx = external_sampling(value_actions, ctx);
  const Action a = value_actions[a_idx];
  if(is_debug) Logger::log("[" + pos_to_str(ctx.state) + "] Applying (external): " + a.to_string());
  auto next_node = next_bp_node(a, ctx.state, ctx.bp_node, ctx.bp_state);
  ctx.state.apply_in_place(a);
  const int branching_idx = value_actions.size() == branching_actions.size() ? a_idx : 0;
  return traverse_mccfr_p(MCCFRContext<StorageT>{ctx.state, next_regret_storage(ctx.regret_storage, branching_idx, ctx.state, ctx.i),
      next_node, next_consec_folds(ctx.consec_folds, a), ctx});
}

template <template<typename> class StorageT>
int MCCFRSolver<StorageT>::traverse_mccfr(const MCCFRContext<StorageT>& ctx) {
  if(is_terminal(ctx.state, ctx.i)) {
    const int u = terminal_utility(ctx);
    if(is_debug) log_utility(u, ctx);
    return u;
  }
  if(ctx.consec_folds >= get_config().restrict_players) {
    return 0;
  }
  if(ctx.state.get_active() == ctx.i) {
    const auto& value_actions = regret_value_actions(ctx.regret_storage);
    const auto& branching_actions = regret_branching_actions(ctx.regret_storage);
    const int n_value_actions = value_actions.size();
    const int cluster = context_cluster(ctx);
    if(is_debug) Logger::log("Cluster: " + std::to_string(cluster));
    std::atomic<int>* base_ptr = get_base_regret_ptr(ctx.regret_storage, cluster);
    int values[MAX_ACTIONS];
    double v_exact = 0.0;
    double v_r_sum = 0.0;
    double v_a_sum = 0.0;
    for(int a_idx = 0; a_idx < n_value_actions; ++a_idx) {
      Action a = value_actions[a_idx];
      if(is_debug) Logger::log("[" + pos_to_str(ctx.state) + "] Applying (traverser): " + a.to_string());
      const int branching_idx = n_value_actions == branching_actions.size() ? a_idx : 0;
      SlimPokerState next_state = ctx.state.apply_copy(a);
      const int v_a = traverse_mccfr(MCCFRContext<StorageT>{next_state, next_regret_storage(ctx.regret_storage, branching_idx, next_state, ctx.i),
        next_bp_node(a, ctx.state, ctx.bp_node, ctx.bp_state), next_consec_folds(ctx.consec_folds, a), ctx});
      const int v_r = std::max(base_ptr[a_idx].load(std::memory_order_relaxed), 0);
      values[a_idx] = v_a;
      v_exact += static_cast<double>(v_r) * static_cast<double>(v_a);
      v_r_sum += v_r;
      v_a_sum += v_a;
      // if(is_debug) log_action_ev(a, freq[a_idx], v_a);
    }
    v_exact = v_r_sum > 0 ? v_exact / v_r_sum : v_a_sum / n_value_actions;
    const int v = static_cast<int>(std::lrint(v_exact));
    if(is_debug) log_net_ev(v, v_exact);
    if(!is_frozen(cluster, ctx.regret_storage)) {
      for(int a_idx = 0; a_idx < n_value_actions; ++a_idx) {
        auto& r_atom = base_ptr[a_idx];
        const int prev_r = r_atom.load(std::memory_order_relaxed);
        const int d_r = values[a_idx] - v;
        const int next_r = prev_r + d_r;
        if(is_debug && next_r > 2'000'000'000) Logger::error("Regret overflowing!\n" + info_str(prev_r, d_r, ctx));
        if(next_r > REGRET_FLOOR) {
          r_atom.fetch_add(d_r, std::memory_order_relaxed);
        }
        if(is_debug) log_regret(value_actions[a_idx], d_r, next_r);
      }
    }
    return v;
  }
  const auto& value_actions = regret_value_actions(ctx.regret_storage);
  const auto& branching_actions = regret_branching_actions(ctx.regret_storage);
  const int a_idx = external_sampling(value_actions, ctx);
  const Action a = value_actions[a_idx];
  if(is_debug) Logger::log("[" + pos_to_str(ctx.state) + "] Applying (external): " + a.to_string());
  auto next_node = next_bp_node(a, ctx.state, ctx.bp_node, ctx.bp_state);
  ctx.state.apply_in_place(a);
  const int branching_idx = value_actions.size() == branching_actions.size() ? a_idx : 0;
  return traverse_mccfr(MCCFRContext<StorageT>{ctx.state, next_regret_storage(ctx.regret_storage, branching_idx, ctx.state, ctx.i),
      next_node, next_consec_folds(ctx.consec_folds, a), ctx});
}

template <template<typename> class StorageT>
int MCCFRSolver<StorageT>::external_sampling(const std::vector<Action>& actions, const MCCFRContext<StorageT>& ctx) {
  const int cluster = context_cluster(ctx);
  const std::atomic<int>* base_ptr = get_base_regret_ptr(ctx.regret_storage, cluster);
  const int a_idx = sample_idx_from_regrets(base_ptr, actions.size());
  // if(is_debug) log_external_sampling(actions[a_idx], actions, freq);
  return a_idx;
}

template <template<typename> class StorageT>
std::string MCCFRSolver<StorageT>::track_wandb_metrics(const long t) const {
  const auto t_i = std::chrono::high_resolution_clock::now();
  nlohmann::json metrics = {};
  metrics["t (M)"] = static_cast<float>(t / 1'000'000.0);
  std::ostringstream out_str;
  out_str << std::setprecision(1) << std::fixed << std::setw(7) << t / 1'000'000.0 << "M it   ";
  track_regret(metrics, out_str, t);
  track_strategy(metrics, out_str);
  const auto t_f = std::chrono::high_resolution_clock::now();
  const auto dt = std::chrono::duration_cast<std::chrono::microseconds>(t_f - t_i).count();
  out_str << std::setw(8) << dt << " us (metrics)";
  Logger::dump(out_str);
  return metrics.dump();
}

bool should_track_strategy(const PokerState& prev_state, const PokerState& next_state, const SolverConfig& solver_config, const MetricsConfig& metrics_config) {
  return prev_state.active_players() > 1 &&
      prev_state.get_round() == solver_config.init_state.get_round() &&
      (next_state.get_round() > 0 || next_state.vpip_players() <= metrics_config.max_vpip) &&
      prev_state.get_bet_level() <= metrics_config.max_bet_level &&
      !should_restrict(prev_state.get_action_history().get_history(), solver_config.restrict_players) &&
      metrics_config.should_track(prev_state);
}

std::string action_label_str(const PokerState& state, const Action a) {
  if(a != Action::CHECK_CALL) return a.to_string();
  const Player& p = state.get_players()[state.get_active()];
  return p.get_betsize() < state.get_max_bet() && p.get_chips() > 0 ? "Call" : "Check";
}

std::string strategy_label(const PokerState& state, const PokerState& init_state, const Action action, const bool phi) {
  const auto rel_actions = state.get_action_history().slice(init_state.get_action_history().size()).get_history();
  std::ostringstream oss;
  oss << pos_to_str(state) << " vs " << static_cast<int>(state.get_bet_level()) << "-bet/";
  PokerState curr_state = init_state;
  for(int a_idx = 0; a_idx < rel_actions.size(); ++a_idx) {
    const Action a = rel_actions[a_idx];
    if(state.has_player_vpip(curr_state.get_active())) {
      oss << pos_to_str(curr_state) << " " << action_label_str(curr_state, a) << ", ";
    }
    curr_state = curr_state.apply(rel_actions[a_idx]);
  }
  oss << "[" << pos_to_str(curr_state) << " " << action.to_string() << "]"
      << (phi ? " (phi)" : " (regrets)");
  return oss.str();
}

template <template<typename> class StorageT>
void MCCFRSolver<StorageT>::track_strategy_by_decision(const PokerState& state, const std::vector<PokerRange>& ranges, 
    const DecisionAlgorithm& decision, const MetricsConfig& metrics_config, const bool phi, nlohmann::json& metrics) const {
  PokerRange base_range = ranges[state.get_active()];
  base_range.remove_cards(get_config().init_board);
  if(state.get_round() >= 4) return;
  for(Action a : valid_actions(state, get_config().action_profile)) {
    PokerState next_state = state.apply(a);
    if(!should_track_strategy(state, next_state, get_config(), metrics_config)) continue;
    if(a == Action::FOLD) {
      track_strategy_by_decision(next_state, ranges, decision, metrics_config, phi, metrics);
    }
    else {
      std::vector<PokerRange> next_ranges = ranges;
      PokerRange action_range = build_action_range(base_range, a, state, Board{get_config().init_board}, decision);
      PokerRange next_range = base_range * action_range;
      const std::string data_label = strategy_label(state, get_config().init_state, a, phi);
      const double n_base_combos = base_range.n_combos();
      metrics[data_label] = n_base_combos > 0.0 ? next_range.n_combos() / n_base_combos : 0.0;
      next_ranges[state.get_active()] = next_range;
      track_strategy_by_decision(next_state, next_ranges, decision, metrics_config, phi, metrics);
    }
  }
}

// to allow use of MCCFRSolver::traverse_mccfr friend in benchmark_mccfr.cpp without moving MCCFRSolver::traverse_mccfr to the header mccfr.hpp 
// (because it's a template and used in a different translation unit)
template class MCCFRSolver<TreeStorageNode>;

// ==========================================================================================
// || TreeSolver
// ==========================================================================================

void TreeSolver::on_start() {
  if(!_regrets_root) {
    Logger::log("Initializing regret storage tree ...");
    _regrets_root = std::make_unique<TreeStorageNode<int>>(get_config().init_state, make_tree_config());
  }
}

std::atomic<int>* TreeSolver::get_base_regret_ptr(TreeStorageNode<int>* storage, const int cluster) {
  return storage->get(cluster); 
}

TreeStorageNode<int>* TreeSolver::init_regret_storage() { 
  return _regrets_root.get();
}

TreeStorageNode<int>* TreeSolver::next_regret_storage(TreeStorageNode<int>* storage, const int action_idx, const SlimPokerState& next_state, const int i) {
  return !is_terminal(next_state, i) ? storage->apply_index(action_idx, next_state) : nullptr;
}

const std::vector<Action>& TreeSolver::regret_branching_actions(TreeStorageNode<int>* storage) const {
  return storage->get_branching_actions();
}

const std::vector<Action>& TreeSolver::regret_value_actions(TreeStorageNode<int>* storage) const {
  return storage->get_value_actions();
}

const std::vector<Action>& TreeSolver::avg_branching_actions(TreeStorageNode<float>* storage) const {
  return storage->get_branching_actions();
}

const std::vector<Action>& TreeSolver::avg_value_actions(TreeStorageNode<float>* storage) const {
  return storage->get_value_actions();
}

// ==========================================================================================
// || BlueprintSolver
// ==========================================================================================

template <template<typename> class StorageT>
BlueprintSolver<StorageT>::BlueprintSolver(const SolverConfig& config, const BlueprintSolverConfig& bp_config)
    : MCCFRSolver<StorageT>{config}, _bp_config{bp_config} {}

template <template<typename> class StorageT>
bool BlueprintSolver<StorageT>::is_update_terminal(const SlimPokerState& state, const int i) const {
  return state.get_winner() != -1 || state.get_round() > 0 || state.get_players()[i].has_folded();
}

template <template<typename> class StorageT>
void BlueprintSolver<StorageT>::update_strategy(const UpdateContext<StorageT>& ctx) {
  if(is_update_terminal(ctx.state, ctx.i)) {
    return;
  }
  if(ctx.consec_folds >= this->get_config().restrict_players) {
    return;
  }
  if(ctx.state.get_active() == ctx.i) {
    const auto& actions = this->avg_value_actions(ctx.avg_storage);
    int cluster = context_cluster(ctx);
    const std::atomic<int>* base_ptr = this->get_base_regret_ptr(ctx.regret_storage, cluster);
    float freq[MAX_ACTIONS];
    calculate_strategy_in_place(base_ptr, actions.size(), freq);
    int a_idx = sample_action_idx(freq, actions.size());
    if(is_debug) {
      Logger::log("Update strategy: " + ctx.hands[ctx.i].to_string() + " (cluster=" + std::to_string(cluster) + ")");
      std::ostringstream debug;
      for(int ai = 0; ai < actions.size(); ++ai) {
        debug << actions[ai].to_string() << "=" << std::setprecision(2) << std::fixed << freq[ai] << "  ";
      }
      Logger::dump(debug);
    }
    this->get_base_avg_ptr(ctx.avg_storage, cluster)[a_idx].fetch_add(1.0f, std::memory_order_relaxed);
    const Action a = actions[a_idx];
    ctx.state.apply_in_place(a);
    update_strategy(UpdateContext<StorageT>{ctx.state, this->next_regret_storage(ctx.regret_storage, a_idx, ctx.state, ctx.i),
                    this->next_avg_storage(ctx.avg_storage, a_idx, ctx.state, ctx.i), next_consec_folds(ctx.consec_folds, a), ctx});
  }
  else {
    const auto& actions = this->avg_branching_actions(ctx.avg_storage);
    for(int a_idx = 0; a_idx < actions.size(); ++a_idx) {
      const Action a = actions[a_idx];
      SlimPokerState next_state = ctx.state.apply_copy(a);
      update_strategy(UpdateContext<StorageT>{next_state, this->next_regret_storage(ctx.regret_storage, a_idx, next_state, ctx.i),
          this->next_avg_storage(ctx.avg_storage, a_idx, next_state, ctx.i), next_consec_folds(ctx.consec_folds, a), ctx});
    }
  }
}

template<template <typename> class StorageT>
void BlueprintSolver<StorageT>::on_start() {
  Logger::log("Blueprint solver config:\n" + _bp_config.to_string());
  Logger::log((BlueprintClusterMap::get_instance() ? "Initialized" : "Failed to initialize") + std::string{" blueprint cluster map."});
}

template <template<typename> class StorageT>
void BlueprintSolver<StorageT>::on_step(const long t, const int i, const std::vector<Hand>& hands, std::vector<CachedIndexer>& indexers) {
  if(t > 0 && t % get_blueprint_config().strategy_interval == 0 && t < get_blueprint_config().preflop_threshold) {
    if(is_debug) Logger::log("============== Updating strategy ==============");
    SlimPokerState state{this->get_config().init_state};
    update_strategy(UpdateContext<StorageT>{state, i, 0, Board{this->get_config().init_board}, hands, indexers,
        this->init_regret_storage(), this->init_avg_storage()});
  }
}

template <template<typename> class StorageT>
bool BlueprintSolver<StorageT>::should_prune(const long t) const {
  return t >= _bp_config.prune_thresh && GSLGlobalRNG::uniform() > 0.95;
}

template <template<typename> class StorageT>
long BlueprintSolver<StorageT>::next_step(const long t, const long T) const {
  return std::min(std::min(_bp_config.next_discount_step(t, T), _bp_config.next_snapshot_step(t, T)), T);
}

template<template<typename> class StorageT>
int BlueprintSolver<StorageT>::get_cluster(const SlimPokerState& state, const Board& board, const std::vector<Hand>& hands, std::vector<CachedIndexer>& indexers) {
  return BlueprintClusterMap::get_instance()->cluster(state.get_round(), indexers[state.get_active()].index(board, hands[state.get_active()], state.get_round()));
}

// ==========================================================================================
// || RealTimeSolver
// ==========================================================================================

template <template<typename> class StorageT>
RealTimeSolver<StorageT>::RealTimeSolver(const std::shared_ptr<const SampledBlueprint>& bp, const RealTimeSolverConfig& rt_config)
    : _bp{bp}, _root_node{bp->get_strategy()->apply(rt_config.init_actions)}, _rt_config{rt_config} {}

template<template <typename> class StorageT>
const StorageT<uint8_t>* RealTimeSolver<StorageT>::next_bp_node(const Action a, const SlimPokerState& state, const StorageT<uint8_t>* bp_node,
    SlimPokerState& bp_state) {
  if(_rt_config.is_terminal() || state.apply_copy(a).is_terminal()) return nullptr;
  if(!is_bias(a) && bp_state.get_round() == state.get_round() && bp_state.get_active() == state.get_active()) {
    // ignore action if bp_state ran ahead (bp_state can never fall behind, only run ahead due to action translation)
    const Action translated = translate_pseudo_harmonic(a, bp_node->get_branching_actions(), state); // TODO: cache by node pointer
    bp_state.apply_in_place(translated);
    return bp_node->apply(a);
  }
  return bp_node;
}

template <template<typename> class StorageT>
Action RealTimeSolver<StorageT>::next_rollout_action(CachedIndexer& indexer, const SlimPokerState& state, const Hand& hand, const Board& board,
    const TreeStorageNode<uint8_t>* node) const {
  const hand_index_t hand_idx = indexer.index(board, hand, state.get_round());
  const int cluster = BlueprintClusterMap::get_instance()->cluster(state.get_round(), hand_idx);
  // std::cout << "Rollout cluster=" << cluster << "\n";
  const uint8_t bias_offset = _bp->bias_offset(state.get_biases()[state.get_active()]);
  // std::cout << "Bias offset=" << static_cast<int>(bias_offset) << "\n";
  const Action action = _bp->decompress_action(node->get(cluster, bias_offset)->load());
  const auto& player = state.get_players()[state.get_active()];
  if(action == Action::FOLD) {
    return !is_action_valid(action, state) ? Action::CHECK_CALL : action;
  }
  if(action.get_bet_type() > 0.0f && total_bet_size(state, action) > player.get_betsize() + player.get_chips()) {
    return is_action_valid(Action::ALL_IN, state) ? Action::ALL_IN : Action::CHECK_CALL;
  }
  return action;
}

template <template<typename> class StorageT>
int RealTimeSolver<StorageT>::terminal_utility(const MCCFRContext<StorageT>& ctx) const {
  // std::cout << "Calculating terminal utility...\n";
  // std::cout << "Biases=" << actions_to_str(ctx.state.get_biases()) << "\n";
  if(ctx.state.has_biases() && ctx.state.get_active() != ctx.state._first_bias) {
    std::ostringstream oss;
    oss << "Active player changed after biasing. Active=" << static_cast<int>(ctx.state.get_active()) << ", First bias="
        << static_cast<int>(ctx.state._first_bias) << ", Biases=";
    for(Action a : ctx.state.get_biases()) oss << a.to_string() << "  ";
    Logger::error(oss.str());
  }
  SlimPokerState curr_state = ctx.state;
  const TreeStorageNode<uint8_t>* node = ctx.bp_node;
  while(!curr_state.is_terminal() && !curr_state.get_players()[ctx.i].has_folded()) {
    if(curr_state.get_round() == ctx.bp_state.get_round() && curr_state.get_active() == ctx.bp_state.get_active()) {
      const Action rollout_action = next_rollout_action(ctx.indexers[curr_state.get_active()], curr_state, ctx.hands[curr_state.get_active()], ctx.board, node);
      curr_state.apply_in_place(rollout_action);
      if(!curr_state.is_terminal()) node = node->apply(rollout_action);
    }
    else {
      // roll state forward until real state and blueprint state are aligned again
      curr_state.apply_in_place(Action::CHECK_CALL);
    }
  }
  return utility(curr_state, ctx.i, ctx.board, ctx.hands, this->get_config().init_chips[ctx.i], this->get_config().rake, ctx.eval);
}

template<template <typename> class StorageT>
bool RealTimeSolver<StorageT>::is_terminal(const SlimPokerState& state, const int i) const {
  return state.has_biases() || state.is_terminal() || state.get_players()[i].has_folded();
}

template<template <typename> class StorageT>
void RealTimeSolver<StorageT>::on_start() {
  Logger::log("Real time solver config:\n" + _rt_config.to_string());
  // Logger::log((RealTimeClusterMap::get_instance() ? "Initialized" : "Failed to initialize") + std::string{" real time cluster map."}); // TODO
}

template<template <typename> class StorageT>
int RealTimeSolver<StorageT>::get_cluster(const SlimPokerState& state, const Board& board, const std::vector<Hand>& hands, std::vector<CachedIndexer>& indexers) {
  if(state.get_round() == this->get_config().init_state.get_round()) return HoleCardIndexer::get_instance()->index(hands[state.get_active()]);
  Logger::error("Real time clusters on future rounds are not implemented.");
}

// ==========================================================================================
// || TreeBlueprintSolver
// ==========================================================================================

std::shared_ptr<const TreeStorageConfig> TreeBlueprintSolver::make_tree_config() const {
  return std::make_shared<TreeStorageConfig>(TreeStorageConfig{
    ClusterSpec{169, 200, 200, 200},
    ActionMode::make_blueprint_mode(get_config().action_profile)
  });
}

TreeBlueprintSolver::TreeBlueprintSolver(const SolverConfig& config, const BlueprintSolverConfig& bp_config) 
    : TreeSolver{config}, MCCFRSolver{config}, BlueprintSolver{config, bp_config} {}

float TreeBlueprintSolver::frequency(const Action action, const PokerState& state, const Board& board, const Hand& hand) const {
  const TreeDecision decision{get_strategy(), get_config().init_state, false};
  return decision.frequency(action, state, board, hand);
}

void TreeBlueprintSolver::on_start() {
  TreeSolver::on_start();
  BlueprintSolver::on_start();
  if(!_phi_root && get_iteration() < get_blueprint_config().preflop_threshold) {
    Logger::log("Initializing avg storage tree...");
    _phi_root = std::make_unique<TreeStorageNode<float>>(get_config().init_state, make_tree_config());
  }
}

void TreeBlueprintSolver::on_snapshot() {
  if(get_iteration() >= get_blueprint_config().preflop_threshold) {
    Logger::log("Reached preflop threshold. Deleting phi...");
    _phi_root = nullptr;
  }
}

std::atomic<float>* TreeBlueprintSolver::get_base_avg_ptr(TreeStorageNode<float>* storage, const int cluster) {
  return storage->get(cluster);
}

TreeStorageNode<float>* TreeBlueprintSolver::init_avg_storage() { 
  return _phi_root.get();
}

TreeStorageNode<float>* TreeBlueprintSolver::next_avg_storage(TreeStorageNode<float>* storage, const int action_idx, const SlimPokerState& next_state,
    const int i) {
  return !is_update_terminal(next_state, i) ? storage->apply_index(action_idx, next_state) : nullptr;
}

void TreeBlueprintSolver::track_strategy(nlohmann::json& metrics, std::ostringstream& out_str) const {
  const auto init_ranges = get_config().init_ranges;
  const TreeDecision decision{get_strategy(), get_config().init_state, false};
  track_strategy_by_decision(get_config().init_state, init_ranges, decision, get_regret_metrics_config(), false, metrics);
  if(_phi_root) {
    track_strategy_by_decision(get_config().init_state, init_ranges, decision, get_avg_metrics_config(), true, metrics);
  }
}

struct NodeMetrics {
  long max_value_sum = 0L;
  long nodes = 0L;
  long values = 0L;

  NodeMetrics& operator+=(const NodeMetrics& other) {
    max_value_sum += other.max_value_sum;
    nodes += other.nodes;
    values += other.values;
    return *this;
  }
};

template <class T>
NodeMetrics collect_node_metrics(const TreeStorageNode<T>* node) {
  NodeMetrics metrics;
  for(int c = 0; c < node->get_n_clusters(); ++c) {
    T max_v = 0L;
    for(int a_idx = 0; a_idx < node->get_value_actions().size(); ++a_idx) {
      max_v = std::max(node->get(c, a_idx)->load(), max_v);
    }
    metrics.max_value_sum += max_v;
  }
  ++metrics.nodes;
  metrics.values = node->get_n_values();
  for(int a_idx = 0; a_idx < node->get_branching_actions().size(); ++a_idx) {
    if(node->is_allocated(a_idx)) metrics += collect_node_metrics(node->apply_index(a_idx));
  }
  return metrics;
}

void TreeBlueprintSolver::track_regret(nlohmann::json& metrics, std::ostringstream& out_str, const long t) const {
  NodeMetrics regret_metrics = collect_node_metrics(get_strategy());
  const long avg_regret = regret_metrics.max_value_sum / t; // should be sum of the maximum regret at each infoset, not sum of all regrets
  const double free_ram = static_cast<double>(get_free_ram()) / 1'000'000'000.0;
  out_str << std::setw(8) << avg_regret << " avg regret   ";
  out_str << std::setw(12) << regret_metrics.nodes << " regret nodes   ";
  out_str << std::setw(12) << regret_metrics.values << " regret values   ";
  out_str << std::setw(8) << std::fixed << std::setprecision(2) << free_ram << " GB free ram   ";
  metrics["avg max regret"] = static_cast<int>(avg_regret);
  metrics["regret_nodes"] = regret_metrics.nodes;
  metrics["regret_values"] = regret_metrics.values;
  metrics["free_ram"] = free_ram;
  if(_phi_root) {
    NodeMetrics phi_metrics = collect_node_metrics(get_phi_root());
    out_str << std::setw(12) << phi_metrics.nodes << " avg nodes   ";
    out_str << std::setw(12) << phi_metrics.values << " avg values   ";
    metrics["avg_nodes"] = phi_metrics.nodes;
    metrics["avg_values"] = phi_metrics.values;
  }
}

void TreeBlueprintSolver::freeze(const std::vector<float>& freq, const Hand& hand, const Board& board, const ActionHistory& history) {
  Logger::error("Freezing is not implemented for TreeBlueprintSolver.");
}

bool TreeBlueprintSolver::operator==(const TreeBlueprintSolver& other) const {
  return TreeSolver::operator==(other) &&
    BlueprintSolver::operator==(other) &&
    ((!_phi_root && !other._phi_root) || (_phi_root && other._phi_root && *_phi_root == *other._phi_root));
}

// ==========================================================================================
// || TreeRealTimeSolver
// ==========================================================================================

TreeRealTimeSolver::TreeRealTimeSolver(const SolverConfig& config, const RealTimeSolverConfig& rt_config, const std::shared_ptr<const SampledBlueprint>& bp)
    : TreeSolver{config}, MCCFRSolver{config}, RealTimeSolver{bp, rt_config} {
  if(config.init_state.get_action_history().size() != rt_config.init_actions.size()) {
    Logger::error("Init state action count does not match mapped action count.\nInit state actions: "
      + actions_to_str(config.init_state.get_action_history().get_history()) + "\nMapped actions: " + actions_to_str(rt_config.init_actions));
  }
}

float TreeRealTimeSolver::frequency(const Action action, const PokerState& state, const Board& board, const Hand& hand) const {
  // const TreeDecision decision{get_strategy(), get_config().init_state, true}; // TODO: use real time clusters
  const TreeDecision decision{get_strategy(), get_config().init_state, false};
  return decision.frequency(action, state, board, hand);
}

void TreeRealTimeSolver::freeze(const std::vector<float>& freq, const Hand& hand, const Board& board, const ActionHistory& history) {
  if(!init_regret_storage()) on_start();
  PokerState state = get_config().init_state;
  const int cluster = BlueprintClusterMap::get_instance()->cluster(state.get_round(), board, hand);
  TreeStorageNode<int>* node = init_regret_storage();
  for(const Action h_a : history.get_history()) {
    state = state.apply(h_a);
    node = node->apply(h_a, state);
  }
  std::vector<int> regrets;
  for(const float f : freq) regrets.push_back(f * 100'000'000);
  node->freeze(regrets, cluster);
}

bool TreeRealTimeSolver::operator==(const TreeRealTimeSolver& other) const {
  return TreeSolver::operator==(other) && RealTimeSolver::operator==(other);
}

void TreeRealTimeSolver::on_start() {
  TreeSolver::on_start();
  RealTimeSolver::on_start();
}

bool TreeRealTimeSolver::is_frozen(const int cluster, const TreeStorageNode<int>* storage) const {
  return storage->is_frozen(cluster);
}

void TreeRealTimeSolver::track_strategy(nlohmann::json& metrics, std::ostringstream& out_str) const {
  const auto init_ranges = get_config().init_ranges;
  // const TreeDecision decision{get_strategy(), get_config().init_state, true}; TODO: use real time clusters
  const TreeDecision decision{get_strategy(), get_config().init_state, false};
  track_strategy_by_decision(get_config().init_state, init_ranges, decision, get_regret_metrics_config(), false, metrics);
}

std::shared_ptr<const TreeStorageConfig> TreeRealTimeSolver::make_tree_config() const {
  std::vector<int> clusters;
  for(int round = 1; round < 4; ++round) {
    clusters.push_back(round == get_config().init_state.get_round() ? MAX_COMBOS : 500);
  }
  return std::make_shared<TreeStorageConfig>(TreeStorageConfig{
    ClusterSpec{169, clusters[0], clusters[1], clusters[2]},
    ActionMode::make_real_time_mode(get_config().action_profile, get_real_time_config())
  });
}

}
