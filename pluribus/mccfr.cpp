#include <algorithm>
#include <atomic>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <omp.h>
#include <sstream>
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

namespace pluribus {

static constexpr int PRUNE_CUTOFF = -300'000'000;
static constexpr int REGRET_FLOOR = -310'000'000;

int utility(const PokerState& state, const int i, const Board& board, const std::vector<Hand>& hands, const int stack_size, const RakeStructure& rake,
    const omp::HandEvaluator& eval) {
  if(state.get_players()[i].has_folded()) {
    return state.get_players()[i].get_chips() - stack_size;
  }
  if(state.get_winner() != -1) {
    return state.get_players()[i].get_chips() - stack_size + (state.get_winner() == i ? state.get_pot() : 0);
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

  long T = _t + t_plus;
  Logger::log("MCCFRSolver --- Initializing HandIndexer...");
  Logger::log(HandIndexer::get_instance() ? "Success." : "Failure.");
  Logger::log("MCCFRSolver --- Initializing FlatClusterMap...");
  Logger::log(BlueprintClusterMap::get_instance() ? "Success." : "Failure.");
  Logger::log("Solver config:\n" + get_config().to_string());
  on_start();

  Logger::log("Training blueprint from " + std::to_string(_t) + " to " + std::to_string(T));
  std::ostringstream buf;
  while(_t < T) {
    long init_t = _t;
    _t = next_step(_t, T); 
    auto interval_start = std::chrono::high_resolution_clock::now();
    buf << std::setprecision(1) << std::fixed << "Next step: " << _t / 1'000'000.0 << "M\n"; 
    Logger::dump(buf);
    #pragma omp parallel for schedule(dynamic, 1)
    for(long t = init_t; t < _t; ++t) {
      thread_local omp::HandEvaluator eval;
      thread_local Deck deck{get_config().init_board};
      thread_local Board board;
      thread_local std::ostringstream debug;
      thread_local MarginalRejectionSampler sampler{get_config().init_ranges, get_config().init_board, get_config().dead_ranges};
      if(_log_level == SolverLogLevel::DEBUG) debug << "============== t = " << t << " ==============\n";
      if(should_log(t)) {
        std::ostringstream metrics_fn;
        metrics_fn << std::setprecision(1) << std::fixed << t / 1'000'000.0 << ".json";
        write_to_file(_metrics_dir / metrics_fn.str(), track_wandb_metrics(t));
      }
      for(int i = 0; i < get_config().poker.n_players; ++i) {
        if(_log_level == SolverLogLevel::DEBUG) debug << "============== i = " << i << " ==============\n";
        std::vector<CachedIndexer> indexers(get_config().poker.n_players);
        RoundSample sample = sampler.sample();
        board = sample_board(get_config().init_board, sample.mask);
        for(int h_idx = 0; h_idx < sample.hands.size(); ++h_idx) {
          indexers[h_idx].index(board, sample.hands[h_idx], 3); // cache indexes
        }
        on_step(t, i, sample.hands, indexers, debug);
        if(should_prune(t)) {
          if(_log_level == SolverLogLevel::DEBUG) debug << "============== Traverse MCCFR-P ==============\n";
          traverse_mccfr_p(get_config().init_state, t, i, board, sample.hands, indexers, eval, init_regret_storage(), debug);
        }
        else {
          if(_log_level == SolverLogLevel::DEBUG) debug << "============== Traverse MCCFR ==============\n";
          traverse_mccfr(get_config().init_state, t, i, board, sample.hands, indexers, eval, init_regret_storage(), debug);
        }
      }
      if(_log_level == SolverLogLevel::DEBUG) {
        write_to_file(_log_dir / ("t" + std::to_string(t) + ".log"), debug.str());
      }
      debug.str("");
    }
    
    auto interval_end = std::chrono::high_resolution_clock::now();
    buf << "Step duration: " << std::chrono::duration_cast<std::chrono::seconds>(interval_end - interval_start).count() << " s.\n";
    Logger::dump(buf);
    if(should_discount(_t)) {
      Logger::log("============== Discounting ==============");
      double d = get_discount_factor(_t);
      buf << std::setprecision(2) << std::fixed << "Discount factor: " << d << "\n";
      Logger::dump(buf);
      init_regret_storage()->lcfr_discount(d);
      if(StorageT<float>* init_avg = init_avg_storage()) init_avg->lcfr_discount(d);
    }
    if(should_snapshot(_t, T)) {
      std::ostringstream fn_stream;
      Logger::log("============== Saving snapshot ==============");
      fn_stream << date_time_str() << "_t" << std::setprecision(1) << std::fixed << _t / 1'000'000.0 << "M.bin";
      save_snapshot((_snapshot_dir / fn_stream.str()).string());
    }
  }

  Logger::log("============== Blueprint training complete ==============");
}

std::string info_str(const PokerState& state, const int prev_r, const int d_r, const long t, const Board& board, const std::vector<Hand>& hands) {
  std::string str = "r=" + std::to_string(prev_r) + " + " + std::to_string(d_r) + "\nt=" + 
         std::to_string(t) + "\nBoard=" + board.to_string() + "\nHands=";
  for(const auto& hand : hands) str += hand.to_string() + "  ";
  str += "\n" + state.get_action_history().to_string() + "\n";
  return str;
}

template <template<typename> class StorageT>
void MCCFRSolver<StorageT>::error(const std::string& msg, const std::ostringstream& debug) const {
  std::string error_msg = msg;
  if(_log_level == SolverLogLevel::DEBUG) {
    const auto debug_dir = _log_dir / ("thread" + std::to_string(omp_get_thread_num()) + ".error");
    write_to_file(debug_dir, debug.str() + "\nRUNTIME ERROR: " + msg);
    error_msg += "\nDebug logs written to " + debug_dir.string();
  }
  Logger::error(error_msg);
}

std::string relative_history_str(const PokerState& state, const PokerState& init_state) {
  return state.get_action_history().slice(init_state.get_action_history().size()).to_string();
}

void log_utility(const int utility, const PokerState& state, const PokerState& init_state, const std::vector<Hand>& hands, std::ostringstream& debug) {
  debug << "Terminal: " << relative_history_str(state, init_state) << "\n";
  debug << "\tHands: ";
  for(int p_idx = 0; p_idx < state.get_players().size(); ++p_idx) {
    debug << hands[p_idx].to_string() << " ";
  }
  debug << "\n";
  debug << "\tu(z) = " << utility << "\n";
}

void log_action_ev(const Action a, const float freq, const int ev, const PokerState& state, const PokerState& init_state, std::ostringstream& debug) {
  debug << "Action EV: " << relative_history_str(state, init_state) << "\n";
  debug << "\tu(" << a.to_string() << ") @ " << std::setprecision(2) << std::fixed << freq << " = " << ev << "\n";
}

void log_net_ev(const int ev, const float ev_exact, const PokerState& state, const PokerState& init_state, std::ostringstream& debug) {
  debug << "Net EV: " << relative_history_str(state, init_state) << "\n";
  debug << "\tu(sigma) = " << std::setprecision(2) << std::fixed << ev << " (exact=" << ev_exact << ")\n";
}

void log_regret(const Action a, const int d_r, const int next_r, std::ostringstream& debug) {
  debug << "\tR(" << a.to_string() << ") = " << d_r << "\n";
  debug << "\tcum R(" << a.to_string() << ") = " << next_r << "\n";
}

void log_external_sampling(const Action sampled, const std::vector<Action>& actions, const std::vector<float>& freq, const PokerState& state,
    const PokerState& init_state, std::ostringstream& debug) {
  debug << "Sampling: " << relative_history_str(state, init_state) << "\n\t";
  for(int a_idx = 0; a_idx < actions.size(); ++a_idx) {
    debug << std::setprecision(2) << std::fixed << actions[a_idx].to_string() << "=" << freq[a_idx] << " ";
  }
  debug << "\n";
  debug << "\tSampled: " << sampled.to_string() << "\n";
}

bool should_restrict(const std::vector<Action>& actions, const int restrict_players) {
  if(actions.size() < restrict_players) return false;
  for(int i = 0; i < restrict_players; ++i) {
    if(actions[i] != Action::FOLD) return false;
  }
  return true;
}

bool is_terminal_call(const Action a, const int i, const PokerState& state) {
  if(a != Action::CHECK_CALL) return false;
  for(int p_idx = 0; p_idx < state.get_players().size(); ++p_idx) {
    if(p_idx != i && !state.get_players()[p_idx].has_folded() && state.get_players()[p_idx].get_chips() > 0) {
      return false;
    }
  }
  return true;
}

template <template<typename> class StorageT>
int MCCFRSolver<StorageT>::traverse_mccfr_p(const PokerState& state, const long t, const int i, const Board& board, const std::vector<Hand>& hands,
    std::vector<CachedIndexer>& indexers, const omp::HandEvaluator& eval, StorageT<int>* regret_storage, std::ostringstream& debug) {
  if(is_terminal(state, i)) {
    const int u = terminal_utility(state, i, board, hands, get_config().stack_size(i), indexers, eval);
    if(_log_level == SolverLogLevel::DEBUG) log_utility(u, state, get_config().init_state, hands, debug);
    return u;
  }
  if(i > get_config().restrict_players - 1 && should_restrict(state.get_action_history().get_history(), get_config().restrict_players)) {
    return 0;
  }
  if(state.get_active() == i) {
    auto value_actions = regret_value_actions(regret_storage, state, get_config().action_profile);
    auto branching_actions = regret_branching_actions(regret_storage, state, get_config().action_profile);
    const int cluster = BlueprintClusterMap::get_instance()->cluster(state.get_round(), indexers[state.get_active()].index(board, hands[i], state.get_round()));
    if(_log_level == SolverLogLevel::DEBUG) debug << "Cluster:" << cluster << "\n";
    std::atomic<int>* base_ptr = get_base_regret_ptr(regret_storage, state, cluster);
    auto freq = calculate_strategy(base_ptr, value_actions.size());

    std::unordered_map<Action, int> values;
    float v_exact = 0;
    for(int a_idx = 0; a_idx < value_actions.size(); ++a_idx) {
      Action a = value_actions[a_idx];
      if(state.get_round() == 3 || a == Action::FOLD || base_ptr[a_idx].load() > PRUNE_CUTOFF || is_terminal_call(a, i, state)) {
        PokerState next_state = state.apply(a);
        const int branching_idx = value_actions.size() == branching_actions.size() ? a_idx : 0;
        int v_a = traverse_mccfr_p(next_state, t, i, board, hands, indexers, eval, next_regret_storage(regret_storage, branching_idx, next_state, i), debug);
        values[a] = v_a;
        v_exact += freq[a_idx] * v_a;
        if(_log_level == SolverLogLevel::DEBUG) log_action_ev(a, freq[a_idx], v_a, state, get_config().init_state, debug);
      }
    }
    const int v = round(v_exact);
    if(_log_level == SolverLogLevel::DEBUG) log_net_ev(v, v_exact, state, get_config().init_state, debug);
    for(int a_idx = 0; a_idx < value_actions.size(); ++a_idx) {
      Action a = value_actions[a_idx];
      if(auto it = values.find(a); it != values.end()) {
        auto& r_atom = base_ptr[a_idx];
        const int prev_r = r_atom.load();
        int d_r = it->second - v;
        int next_r = prev_r + d_r;
        if(next_r > 2'000'000'000) error("Regret overflowing!\n" + info_str(state, prev_r, d_r, t, board, hands), debug);
        if(next_r > REGRET_FLOOR) {
          r_atom.fetch_add(d_r);
        }
        if(_log_level == SolverLogLevel::DEBUG) log_regret(value_actions[a_idx], d_r, next_r, debug);
      }
    }
    return v;
  }
  auto value_actions = regret_value_actions(regret_storage, state, get_config().action_profile);
  auto branching_actions = regret_branching_actions(regret_storage, state, get_config().action_profile);
  const int a_idx = external_sampling(value_actions, state, board, hands, indexers, regret_storage, debug);
  const PokerState next_state = state.apply(value_actions[a_idx]);
  const int branching_idx = value_actions.size() == branching_actions.size() ? a_idx : 0;
  return traverse_mccfr(next_state, t, i, board, hands, indexers, eval, next_regret_storage(regret_storage, branching_idx, next_state, i), debug);
}

template <template<typename> class StorageT>
int MCCFRSolver<StorageT>::traverse_mccfr(const PokerState& state, const long t, const int i, const Board& board, const std::vector<Hand>& hands,
    std::vector<CachedIndexer>& indexers, const omp::HandEvaluator& eval, StorageT<int>* regret_storage, std::ostringstream& debug) {
  if(is_terminal(state, i)) {
    const int u = terminal_utility(state, i, board, hands, get_config().stack_size(i), indexers, eval);
    if(_log_level == SolverLogLevel::DEBUG) log_utility(u, state, get_config().init_state, hands, debug);
    return u;
  }
  if(i > get_config().restrict_players - 1 && should_restrict(state.get_action_history().get_history(), get_config().restrict_players)) {
    return 0;
  }
  if(state.get_active() == i) {
    auto value_actions = regret_value_actions(regret_storage, state, get_config().action_profile);
    auto branching_actions = regret_branching_actions(regret_storage, state, get_config().action_profile);
    const int cluster = BlueprintClusterMap::get_instance()->cluster(state.get_round(), indexers[state.get_active()].index(board, hands[i], state.get_round()));
    if(_log_level == SolverLogLevel::DEBUG) debug << "Cluster:" << cluster << "\n";
    std::atomic<int>* base_ptr = get_base_regret_ptr(regret_storage, state, cluster);
    auto freq = calculate_strategy(base_ptr, value_actions.size());
    std::unordered_map<Action, int> values;
    float v_exact = 0;
    for(int a_idx = 0; a_idx < value_actions.size(); ++a_idx) {
      Action a = value_actions[a_idx];
      const int branching_idx = value_actions.size() == branching_actions.size() ? a_idx : 0;
      PokerState next_state = state.apply(a);
      int v_a = traverse_mccfr(next_state, t, i, board, hands, indexers, eval, next_regret_storage(regret_storage, branching_idx, next_state, i), debug);
      values[a] = v_a;
      v_exact += freq[a_idx] * v_a;
      if(_log_level == SolverLogLevel::DEBUG) log_action_ev(a, freq[a_idx], v_a, state, get_config().init_state, debug);
    }
    int v = round(v_exact);
    if(_log_level == SolverLogLevel::DEBUG) log_net_ev(v, v_exact, state, get_config().init_state, debug);
    for(int a_idx = 0; a_idx < value_actions.size(); ++a_idx) {
      auto& r_atom = base_ptr[a_idx];
      const int prev_r = r_atom.load();
      int d_r = values[value_actions[a_idx]] - v;
      int next_r = prev_r + d_r;
      if(next_r > 2'000'000'000) error("Regret overflowing!\n" + info_str(state, prev_r, d_r, t, board, hands), debug);
      if(next_r > REGRET_FLOOR) {
        r_atom.fetch_add(d_r);
      }
      if(_log_level == SolverLogLevel::DEBUG) log_regret(value_actions[a_idx], d_r, next_r, debug);
    }
    return v;
  }
  auto value_actions = regret_value_actions(regret_storage, state, get_config().action_profile);
  auto branching_actions = regret_branching_actions(regret_storage, state, get_config().action_profile);
  const int a_idx = external_sampling(value_actions, state, board, hands, indexers, regret_storage, debug);
  const PokerState next_state = state.apply(value_actions[a_idx]);
  const int branching_idx = value_actions.size() == branching_actions.size() ? a_idx : 0;
  return traverse_mccfr(next_state, t, i, board, hands, indexers, eval, next_regret_storage(regret_storage, branching_idx, next_state, i), debug);
}

template <template<typename> class StorageT>
int MCCFRSolver<StorageT>::external_sampling(const std::vector<Action>& actions, const PokerState& state, const Board& board,
    const std::vector<Hand>& hands, std::vector<CachedIndexer>& indexers, StorageT<int>* regret_storage, std::ostringstream& debug) {
  const int cluster = BlueprintClusterMap::get_instance()->cluster(state.get_round(), indexers[state.get_active()].index(board, hands[state.get_active()],
      state.get_round()));
  const std::atomic<int>* base_ptr = get_base_regret_ptr(regret_storage, state, cluster);
  const std::vector<float> freq = calculate_strategy(base_ptr, actions.size());
  const int a_idx = sample_action_idx(freq);
  if(_log_level == SolverLogLevel::DEBUG) log_external_sampling(actions[a_idx], actions, freq, state, get_config().init_state, debug);
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

bool should_track_strategy(const PokerState& state, const SolverConfig& solver_config, const MetricsConfig& metrics_config) {
  return state.active_players() > 1 &&
      state.get_round() == solver_config.init_state.get_round() &&
      (state.get_round() > 0 || state.vpip_players() - (state.has_player_vpip(state.get_active()) ? 1 : 0) <= metrics_config.max_vpip) &&
      state.get_bet_level() <= metrics_config.max_bet_level &&
      !should_restrict(state.get_action_history().get_history(), solver_config.restrict_players) &&
      metrics_config.should_track(state);
}

std::string strategy_label(const PokerState& state, const PokerState& init_state, const Action action, const bool phi) {
  const auto rel_actions = state.get_action_history().slice(init_state.get_action_history().size()).get_history();
  std::ostringstream oss;
  oss << pos_to_str(state) << " vs " << static_cast<int>(state.get_bet_level()) << "-bet/";
  PokerState curr_state = init_state;
  for(int a_idx = 0; a_idx < rel_actions.size(); ++a_idx) {
    if(state.has_player_vpip(curr_state.get_active())) {
      oss << pos_to_str(curr_state) << " " << rel_actions[a_idx].to_string()
        << ", ";
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
  if(!should_track_strategy(state, get_config(), metrics_config)) return;
  PokerRange base_range = ranges[state.get_active()];
  base_range.remove_cards(get_config().init_board);
  for(Action a : valid_actions(state, get_config().action_profile)) {
    PokerRange action_range = build_action_range(base_range, a, state, Board{get_config().init_board}, decision);
    PokerRange next_range = base_range * action_range;
    const std::string data_label = strategy_label(state, get_config().init_state, a, phi);
    metrics[data_label] = next_range.n_combos() / base_range.n_combos();
    std::vector<PokerRange> next_ranges = ranges;
    next_ranges[state.get_active()] = next_range;
    track_strategy_by_decision(state.apply(a), next_ranges, decision, metrics_config, phi, metrics);
  }
}

// to allow use of MCCFRSolver::traverse_mccfr friend in benchmark_mccfr.cpp without moving MCCFRSolver::traverse_mccfr to the header mccfr.hpp 
// (because it's a template and used in a different translation unit)
template class MCCFRSolver<TreeStorageNode>;

// ==========================================================================================
// || TreeSolver
// ==========================================================================================

float TreeSolver::frequency(const Action action, const PokerState& state, const Board& board, const Hand& hand) const {
  const TreeDecision decision{_regrets_root.get(), get_config().init_state};
  return decision.frequency(action, state, board, hand);
}

void TreeSolver::on_start() {
  if(!_regrets_root) {
    Logger::log("Initializing regret storage tree ...");
    _regrets_root = std::make_unique<TreeStorageNode<int>>(get_config().init_state, make_tree_config());
  }
}

std::atomic<int>* TreeSolver::get_base_regret_ptr(TreeStorageNode<int>* storage, const PokerState& state, const int cluster) {
  return storage->get(cluster); 
}

TreeStorageNode<int>* TreeSolver::init_regret_storage() { 
  return _regrets_root.get();
}

TreeStorageNode<int>* TreeSolver::next_regret_storage(TreeStorageNode<int>* storage, const int action_idx, const PokerState& next_state, const int i) {
  return !is_terminal(next_state, i) ? storage->apply_index(action_idx, next_state) : nullptr;
}

std::vector<Action> TreeSolver::regret_branching_actions(TreeStorageNode<int>* storage, const PokerState& state, const ActionProfile& profile) const {
  return storage->get_branching_actions();
}

std::vector<Action> TreeSolver::regret_value_actions(TreeStorageNode<int>* storage, const PokerState& state, const ActionProfile& profile) const {
  return storage->get_value_actions();
}

std::vector<Action> TreeSolver::avg_branching_actions(TreeStorageNode<float>* storage, const PokerState& state, const ActionProfile& profile) const {
  return storage->get_branching_actions();
}

std::vector<Action> TreeSolver::avg_value_actions(TreeStorageNode<float>* storage, const PokerState& state, const ActionProfile& profile) const {
  return storage->get_value_actions();
}

void TreeSolver::track_strategy(nlohmann::json& metrics, std::ostringstream& out_str) const {
  const auto init_ranges = get_config().init_ranges;
  track_strategy_by_decision(get_config().init_state, init_ranges, TreeDecision{_regrets_root.get(), get_config().init_state},
      get_regret_metrics_config(), false, metrics);
}

// ==========================================================================================
// || BlueprintSolver
// ==========================================================================================

template <template<typename> class StorageT>
BlueprintSolver<StorageT>::BlueprintSolver(const SolverConfig& config, const BlueprintSolverConfig& bp_config)
    : MCCFRSolver<StorageT>{config}, _bp_config{bp_config} {}

template <template<typename> class StorageT>
bool BlueprintSolver<StorageT>::is_update_terminal(const PokerState& state, const int i) const {
  return state.get_winner() != -1 || state.get_round() > 0 || state.get_players()[i].has_folded();
}

template <template<typename> class StorageT>
void BlueprintSolver<StorageT>::update_strategy(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, 
    std::vector<CachedIndexer>& indexers, StorageT<int>* regret_storage, StorageT<float>* avg_storage, std::ostringstream& debug) {
  if(is_update_terminal(state, i)) {
    return;
  }
  if(state.get_active() == i) {
    auto actions = this->avg_value_actions(avg_storage, state, this->get_config().action_profile);
    int cluster = BlueprintClusterMap::get_instance()->cluster(state.get_round(), indexers[state.get_active()].index(board, hands[i], state.get_round()));
    const std::atomic<int>* base_ptr = this->get_base_regret_ptr(regret_storage, state, cluster);
    auto freq = calculate_strategy(base_ptr, actions.size());
    int a_idx = sample_action_idx(freq);
    if(this->get_log_level() == SolverLogLevel::DEBUG) {
      debug << "Update strategy: " << relative_history_str(state, this->get_config().init_state) << "\n";
      debug << "\t" << hands[i].to_string() << ": (cluster=" << cluster << ")\n\t";
      for(int ai = 0; ai < actions.size(); ++ai) {
        debug << actions[ai].to_string() << "=" << std::setprecision(2) << std::fixed << freq[ai] << "  ";
      }
      debug << "\n";
    }
    this->get_base_avg_ptr(avg_storage, state, cluster)[a_idx].fetch_add(1.0f);
    PokerState next_state = state.apply(actions[a_idx]);
    update_strategy(next_state, i, board, hands, indexers, this->next_regret_storage(regret_storage, a_idx, next_state, i),
                    this->next_avg_storage(avg_storage, a_idx, next_state, i), debug);
  }
  else {
    auto actions = this->avg_branching_actions(avg_storage, state, this->get_config().action_profile);
    for(int a_idx = 0; a_idx < actions.size(); ++a_idx) {
      PokerState next_state = state.apply(actions[a_idx]);
      update_strategy(next_state, i, board, hands, indexers, this->next_regret_storage(regret_storage, a_idx, next_state, i),
                      this->next_avg_storage(avg_storage, a_idx, next_state, i), debug);
    }
  }
}

template <template<typename> class StorageT>
void BlueprintSolver<StorageT>::on_step(const long t, const int i, const std::vector<Hand>& hands, std::vector<CachedIndexer>& indexers,
    std::ostringstream& debug) {
  if(t > 0 && t % get_blueprint_config().strategy_interval == 0 && t < get_blueprint_config().preflop_threshold) {
    if(this->get_log_level() == SolverLogLevel::DEBUG) debug << "============== Updating strategy ==============\n";
    update_strategy(this->get_config().init_state, i, Board{this->get_config().init_board}, hands, indexers,
        this->init_regret_storage(), this->init_avg_storage(), debug);
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
int BlueprintSolver<StorageT>::get_cluster(const PokerState& state, const Board& board, const std::vector<Hand>& hands, std::vector<CachedIndexer>& indexers) {
  return BlueprintClusterMap::get_instance()->cluster(state.get_round(), indexers[state.get_active()].index(board, hands[state.get_active()], state.get_round()));
}

// ==========================================================================================
// || RealTimeSolver
// ==========================================================================================

template <template<typename> class StorageT>
RealTimeSolver<StorageT>::RealTimeSolver(const std::shared_ptr<const SampledBlueprint>& bp, const RealTimeSolverConfig& rt_config)
    : _bp{bp}, _root_node{bp->get_strategy()->apply(rt_config.init_actions)}, _rt_config{rt_config} {}

template <template<typename> class StorageT>
Action RealTimeSolver<StorageT>::next_rollout_action(CachedIndexer& indexer, const PokerState& state, const Hand& hand, const Board& board) const {
  const hand_index_t hand_idx = indexer.index(board, hand, state.get_round());
  const int cluster = BlueprintClusterMap::get_instance()->cluster(state.get_round(), hand_idx);
  // std::cout << "Rollout cluster=" << cluster << "\n";
  const std::vector<Action> history = state.get_action_history().slice(_rt_config.init_actions.size()).get_history();
  // std::cout << "Live history=" << actions_to_str(history) << "\n";
  // TODO: action translation
  const TreeStorageNode<uint8_t>* node = _root_node->apply(history);
  // std::cout << "Applied!\n";
  const uint8_t bias_offset = _bp->bias_offset(state.get_biases()[state.get_active()]);
  // std::cout << "Bias offset=" << static_cast<int>(bias_offset) << "\n";
  return _bp->decompress_action(node->get(cluster, bias_offset)->load());
}

template <template<typename> class StorageT>
int RealTimeSolver<StorageT>::terminal_utility(const PokerState& state, const int i, const Board& board, const std::vector<Hand>& hands, const int stack_size,
    std::vector<CachedIndexer>& indexers, const omp::HandEvaluator& eval) const {
  if(state.has_biases() && state.get_active() != state._first_bias) {
    std::ostringstream oss;
    oss << "Active player changed after biasing. Active=" << static_cast<int>(state.get_active()) << ", First bias=" << static_cast<int>(state._first_bias)
        << ", Biases=";
    for(Action a : state.get_biases()) oss << a.to_string() << "  ";
    Logger::error(oss.str());
  }
  PokerState curr_state = state;
  while(!curr_state.is_terminal() && !curr_state.get_players()[i].has_folded()) {
    curr_state = curr_state.apply(next_rollout_action(indexers[curr_state.get_active()], curr_state, hands[curr_state.get_active()], board));
  }
  return utility(curr_state, i, board, hands, stack_size, this->get_config().rake, eval);
}

template<template <typename> class StorageT>
int RealTimeSolver<StorageT>::get_cluster(const PokerState& state, const Board& board, const std::vector<Hand>& hands, std::vector<CachedIndexer>& indexers) {
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

void TreeBlueprintSolver::on_start() {
  TreeSolver::on_start();
  BlueprintSolver::on_start();
  if(!_phi_root) {
    Logger::log("Initializing avg storage tree ...");
    _phi_root = std::make_unique<TreeStorageNode<float>>(get_config().init_state, make_tree_config());
  }
}

std::atomic<float>* TreeBlueprintSolver::get_base_avg_ptr(TreeStorageNode<float>* storage, const PokerState& state, const int cluster) {
  return storage->get(cluster);
}

TreeStorageNode<float>* TreeBlueprintSolver::init_avg_storage() { 
  return _phi_root.get();
}

TreeStorageNode<float>* TreeBlueprintSolver::next_avg_storage(TreeStorageNode<float>* storage, const int action_idx, const PokerState& next_state,
    const int i) {
  return !is_update_terminal(next_state, i) ? storage->apply_index(action_idx, next_state) : nullptr;
}

void TreeBlueprintSolver::track_strategy(nlohmann::json& metrics, std::ostringstream& out_str) const {
  TreeSolver::track_strategy(metrics, out_str);
  const auto init_ranges = get_config().init_ranges;
  track_strategy_by_decision(get_config().init_state, init_ranges, TreeDecision{_phi_root.get(), get_config().init_state},
      get_avg_metrics_config(), true, metrics);
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
  NodeMetrics phi_metrics = collect_node_metrics(get_phi_root());
  const long avg_regret = regret_metrics.max_value_sum / t; // should be sum of the maximum regret at each infoset, not sum of all regrets
  const double free_ram = static_cast<double>(get_free_ram()) / 1'000'000'000.0;
  out_str << std::setw(8) << avg_regret << " avg regret   ";
  out_str << std::setw(12) << regret_metrics.nodes << " regret nodes   ";
  out_str << std::setw(12) << regret_metrics.values << " regret values   ";
  out_str << std::setw(12) << phi_metrics.nodes << " avg nodes   ";
  out_str << std::setw(12) << phi_metrics.values << " avg values   ";
  out_str << std::setw(8) << std::fixed << std::setprecision(2) << free_ram << " GB free ram   ";
  metrics["avg max regret"] = static_cast<int>(avg_regret);
  metrics["regret_nodes"] = regret_metrics.nodes;
  metrics["regret_values"] = regret_metrics.values;
  metrics["avg_nodes"] = phi_metrics.nodes;
  metrics["avg_values"] = phi_metrics.values;
  metrics["free_ram"] = free_ram;
}

bool TreeBlueprintSolver::operator==(const TreeBlueprintSolver& other) const {
  return TreeSolver::operator==(other) && BlueprintSolver::operator==(other) && *_phi_root == *other._phi_root;
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

bool TreeRealTimeSolver::operator==(const TreeRealTimeSolver& other) const {
  return TreeSolver::operator==(other) && RealTimeSolver::operator==(other);
}

void TreeRealTimeSolver::on_start() {
  TreeSolver::on_start();
  RealTimeSolver::on_start();
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