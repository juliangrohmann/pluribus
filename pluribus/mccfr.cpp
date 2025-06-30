#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <atomic>
#include <limits>
#include <omp.h>
#include <tqdm/tqdm.hpp>
#include <json/json.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/unordered_map.hpp>
#include <pluribus/util.hpp>
#include <pluribus/rng.hpp>
#include <pluribus/debug.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/cluster.hpp>
#include <pluribus/actions.hpp>
#include <pluribus/traverse.hpp>
#include <pluribus/decision.hpp>
#include <pluribus/logging.hpp>
#include <pluribus/mccfr.hpp>

namespace pluribus {

static const int PRUNE_CUTOFF = -300'000'000;
static const int REGRET_FLOOR = -310'000'000;

int utility(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, int stack_size, 
    const omp::HandEvaluator& eval) {
  if(state.get_players()[i].has_folded()) {
    return state.get_players()[i].get_chips() - stack_size;
  }
  else if(state.get_winner() != -1) {
    return state.get_players()[i].get_chips() - stack_size + (state.get_winner() == i ? state.get_pot() : 0);
  }
  else if(state.get_round() >= 4) {
    return state.get_players()[i].get_chips() - stack_size + showdown_payoff(state, i, board, hands, eval);
  }
  else {
    Logger::error("Non-terminal state does not have utility.");
    return -1;
  }
}

Solver::Solver(const SolverConfig& config) : _config{config} {
  if(config.init_board.size() != n_board_cards(config.init_state.get_round())) {
    Logger::error("Wrong amount of solver board cards. Round=" + round_to_str(config.init_state.get_round()) + 
        ", Board=" + cards_to_str(config.init_board));
  }
  if(config.init_state.get_players().size() != config.poker.n_players) Logger::error("Player number mismatch in Solver.");
}

void Solver::solve(long t_plus) {
  Logger::log("================================= Solve ==================================");
  _state = SolverState::SOLVING;
  _solve(t_plus);
  _state = SolverState::SOLVED;
}

// ==========================================================================================
// || MCCFRSolver
// ==========================================================================================

std::string action_str(const std::vector<Action>& actions) {
  std::string str;
  for(Action a : actions) {
    str += a.to_string() + ", ";
  }
  return str;
}

bool are_full_ranges(const std::vector<PokerRange>& ranges) {
  PokerRange full_range = PokerRange::full();
  for(const auto& r : ranges) {
    if(r != full_range) return false;
  }
  return true;
}

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
  Logger::log(FlatClusterMap::get_instance() ? "Success." : "Failure.");
  Logger::log("Solver config:\n" + get_config().to_string());
  on_start();

  // omp_set_num_threads(1);
  bool full_ranges = are_full_ranges(get_config().init_ranges);
  Logger::log("Full ranges: " + std::string{full_ranges ? "true" : "false"});
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
      thread_local std::vector<Hand> hands{static_cast<size_t>(get_config().poker.n_players)};
      thread_local std::ostringstream debug;
      if(_log_level == SolverLogLevel::DEBUG) debug << "============== t = " << t << " ==============\n";
      // if(t % (_config.log_interval) == 0) log_metrics(t);
      if(should_log(t)) {
        std::ostringstream metrics_fn;
        metrics_fn << std::setprecision(1) << std::fixed << t / 1'000'000.0 << ".json";
        write_to_file(_metrics_dir / metrics_fn.str(), track_wandb_metrics(t));
      }
      for(int i = 0; i < get_config().poker.n_players; ++i) {
        if(_log_level == SolverLogLevel::DEBUG) debug << "============== i = " << i << " ==============\n";
        deck.shuffle();
        board.deal(deck, get_config().init_board);
        std::vector<CachedIndexer> indexers(get_config().poker.n_players);
        if(full_ranges) {
          for(int h_idx = 0; h_idx < hands.size(); ++h_idx) { 
            hands[h_idx].deal(deck);
            indexers[h_idx].index(board, hands[h_idx], 3); // cache indexes
          }
        }
        else {
          std::unordered_set<uint8_t> dead_cards;
          std::copy(board.cards().begin(), board.cards().end(), std::inserter(dead_cards, dead_cards.end()));
          for(int p_idx = 0; p_idx < get_config().poker.n_players; ++p_idx) {
            Logger::error("Biased MCCFR sampling not implemented.");
            // hands[p_idx] = _config.init_ranges[p_idx].sample(dead_cards);
            dead_cards.insert(hands[p_idx].cards()[0]);
            dead_cards.insert(hands[p_idx].cards()[1]);
          }
        }

        on_step(t, i, hands, debug);
        // if(t > _config.prune_thresh) {
        if(should_prune(t)) {
          if(_log_level == SolverLogLevel::DEBUG) debug << "============== Traverse MCCFR-P ==============\n";
          traverse_mccfr_p(get_config().init_state, t, i, board, hands, indexers, eval, init_regret_storage(), init_avg_storage(), debug);
        }
        else {
          if(_log_level == SolverLogLevel::DEBUG) debug << "============== Traverse MCCFR ==============\n";
          traverse_mccfr(get_config().init_state, t, i, board, hands, indexers, eval, init_regret_storage(), init_avg_storage(), debug);
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
    // if(_t == next_discount) {
    if(should_discount(_t)) {
      Logger::log("============== Discounting ==============");
      // long discount_interval = _config.discount_interval;
      // double d = static_cast<double>(_t / discount_interval) / (_t / discount_interval + 1);
      double d = get_discount_factor(_t);
      buf << std::setprecision(2) << std::fixed << "Discount factor: " << d << "\n";
      Logger::dump(buf);
      init_regret_storage()->lcfr_discount(d);
      StorageT<float>* init_avg = init_avg_storage();
      if(init_avg) init_avg->lcfr_discount(d);
    }
    if(should_snapshot(_t, T)) {
      std::ostringstream fn_stream;
      Logger::log("============== Saving snapshot ==============");
      fn_stream << date_time_str() << "_t" << std::setprecision(1) << std::fixed << _t / 1'000'000.0 << "M.bin";
      cereal_save(*this, (_snapshot_dir / fn_stream.str()).string());
    }
  }

  Logger::log("============== Blueprint training complete ==============");
}

std::string info_str(const PokerState& state, int prev_r, int d_r, long t, const Board& board, const std::vector<Hand>& hands) {
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
    auto debug_dir = _log_dir / ("thread" + std::to_string(omp_get_thread_num()) + ".error");
    write_to_file(debug_dir, debug.str() + "\nRUNTIME ERROR: " + msg);
    error_msg += "\nDebug logs written to " + debug_dir.string();
  }
  Logger::error(error_msg);
}

std::string relative_history_str(const PokerState& state, const PokerState& init_state) {
  return state.get_action_history().slice(init_state.get_action_history().size()).to_string();
}

void log_utility(int utility, const PokerState& state, const PokerState& init_state, const std::vector<Hand>& hands, std::ostringstream& debug) {
  debug << "Terminal: " << relative_history_str(state, init_state) << "\n";
  debug << "\tHands: ";
  for(int p_idx = 0; p_idx < state.get_players().size(); ++p_idx) {
    debug << hands[p_idx].to_string() << " ";
  }
  debug << "\n";
  debug << "\tu(z) = " << utility << "\n";
}

void log_action_ev(Action a, float freq, int ev, const PokerState& state, const PokerState& init_state, std::ostringstream& debug) {
  debug << "Action EV: " << relative_history_str(state, init_state) << "\n";
  debug << "\tu(" << a.to_string() << ") @ " << std::setprecision(2) << std::fixed << freq << " = " << ev << "\n";
}

void log_net_ev(bool frozen, int ev, float ev_exact, const PokerState& state, const PokerState& init_state, std::ostringstream& debug) {
  debug << "Net EV: " << relative_history_str(state, init_state) << "\n";
  debug << "\tu(sigma) = " << std::setprecision(2) << std::fixed << ev << " (exact=" << ev_exact << ")\n";
  if(state.get_round() == 0 && frozen) debug << "Preflop frozen, skipping regret update.\n";
}

void log_regret(Action a, int d_r, int next_r, std::ostringstream& debug) {
  debug << "\tR(" << a.to_string() << ") = " << d_r << "\n";
  debug << "\tcum R(" << a.to_string() << ") = " << next_r << "\n";
}

void log_external_sampling(Action sampled, const std::vector<Action>& actions, const std::vector<float>& freq, const PokerState& state, 
    const PokerState& init_state, std::ostringstream& debug) {
  debug << "Sampling: " << relative_history_str(state, init_state) << "\n\t";
  for(int a_idx = 0; a_idx < actions.size(); ++a_idx) {
    debug << std::setprecision(2) << std::fixed << actions[a_idx].to_string() << "=" << freq[a_idx] << " ";
  }
  debug << "\n";
  debug << "\tSampled: " << sampled.to_string() << "\n";
}

template <template<typename> class StorageT>
int MCCFRSolver<StorageT>::traverse_mccfr_p(const PokerState& state, long t, int i, const Board& board, const std::vector<Hand>& hands, 
    std::vector<CachedIndexer>& indexers, const omp::HandEvaluator& eval, StorageT<int>* regret_storage, StorageT<float>* avg_storage, 
    std::ostringstream& debug) {
  if(is_terminal(state, i)) {
    int u = terminal_utility(state, i, board, hands, get_config().poker.n_chips, indexers, eval);
    if(_log_level == SolverLogLevel::DEBUG) log_utility(u, state, get_config().init_state, hands, debug);
    return u;
  }
  else if(state.get_active() == i) {
    auto actions = regret_node_actions(regret_storage, state, get_config().action_profile);
    int cluster = FlatClusterMap::get_instance()->cluster(state.get_round(), indexers[state.get_active()].index(board, hands[i], state.get_round()));
    if(_log_level == SolverLogLevel::DEBUG) debug << "Cluster:" << cluster << "\n";
    std::atomic<int>* base_ptr = get_base_regret_ptr(regret_storage, state, cluster);
    auto freq = calculate_strategy(base_ptr, actions.size());

    std::unordered_map<Action, int> values;
    float v_exact = 0;
    for(int a_idx = 0; a_idx < actions.size(); ++a_idx) {
      Action a = actions[a_idx];
      if(base_ptr[a_idx].load() > PRUNE_CUTOFF) {
        PokerState next_state = state.apply(a);
        int v_a = traverse_mccfr_p(next_state, t, i, board, hands, indexers, eval, next_regret_storage(regret_storage, a_idx, next_state, i), 
            next_avg_storage(avg_storage, a_idx, next_state, i), debug);
        values[a] = v_a;
        v_exact += freq[a_idx] * v_a;
        if(_log_level == SolverLogLevel::DEBUG) log_action_ev(a, freq[a_idx], v_a, state, get_config().init_state, debug);
      }
    }
    int v = round(v_exact);
    if(_log_level == SolverLogLevel::DEBUG) log_net_ev(is_preflop_frozen(t), v, v_exact, state, get_config().init_state, debug);
    if(state.get_round() == 0 && is_preflop_frozen(t)) return v;
    for(int a_idx = 0; a_idx < actions.size(); ++a_idx) {
      Action a = actions[a_idx];
      auto it = values.find(a);
      if(it != values.end()) {
        auto& r_atom = base_ptr[a_idx];
        int prev_r = r_atom.load();
        int d_r = (*it).second - v;
        int next_r = prev_r + d_r;
        if(next_r > 2'000'000'000) error("Regret overflowing!\n" + info_str(state, prev_r, d_r, t, board, hands), debug);
        if(next_r > REGRET_FLOOR) {
          r_atom.fetch_add(d_r);
        }
        if(_log_level == SolverLogLevel::DEBUG) log_regret(actions[a_idx], d_r, next_r, debug);
      }
    }
    return v;
  }
  else {
    auto actions = regret_node_actions(regret_storage, state, get_config().action_profile);
    int a_idx = external_sampling(actions, state, t, board, hands, indexers, eval, regret_storage, avg_storage, debug);
    PokerState next_state = state.apply(actions[a_idx]);
    return traverse_mccfr_p(next_state, t, i, board, hands, indexers, eval, next_regret_storage(regret_storage, a_idx, next_state, i), 
        next_avg_storage(avg_storage, a_idx, next_state, i), debug);
  }
}

template <template<typename> class StorageT>
int MCCFRSolver<StorageT>::traverse_mccfr(const PokerState& state, long t, int i, const Board& board, const std::vector<Hand>& hands, 
    std::vector<CachedIndexer>& indexers, const omp::HandEvaluator& eval, StorageT<int>* regret_storage, StorageT<float>* avg_storage, 
    std::ostringstream& debug) {
  if(is_terminal(state, i)) {
    int u = terminal_utility(state, i, board, hands, get_config().poker.n_chips, indexers, eval);
    if(_log_level == SolverLogLevel::DEBUG) log_utility(u, state, get_config().init_state, hands, debug);
    return u;
  }
  else if(state.get_active() == i) {
    auto actions = regret_node_actions(regret_storage, state, get_config().action_profile);
    int cluster = FlatClusterMap::get_instance()->cluster(state.get_round(), indexers[state.get_active()].index(board, hands[i], state.get_round()));
    if(_log_level == SolverLogLevel::DEBUG) debug << "Cluster:" << cluster << "\n";
    std::atomic<int>* base_ptr = get_base_regret_ptr(regret_storage, state, cluster);
    auto freq = calculate_strategy(base_ptr, actions.size());
    std::unordered_map<Action, int> values;
    float v_exact = 0;
    for(int a_idx = 0; a_idx < actions.size(); ++a_idx) {
      Action a = actions[a_idx];
      PokerState next_state = state.apply(a);
      int v_a = traverse_mccfr(next_state, t, i, board, hands, indexers, eval, next_regret_storage(regret_storage, a_idx, next_state, i), 
          next_avg_storage(avg_storage, a_idx, next_state, i), debug);
      values[a] = v_a;
      v_exact += freq[a_idx] * v_a;
      if(_log_level == SolverLogLevel::DEBUG) log_action_ev(a, freq[a_idx], v_a, state, get_config().init_state, debug);
    }
    int v = round(v_exact);
    if(_log_level == SolverLogLevel::DEBUG) log_net_ev(is_preflop_frozen(t), v, v_exact, state, get_config().init_state, debug);
    if(state.get_round() == 0 && is_preflop_frozen(t)) return v;

    for(int a_idx = 0; a_idx < actions.size(); ++a_idx) {
      auto& r_atom = base_ptr[a_idx];
      int prev_r = r_atom.load();
      int d_r = values[actions[a_idx]] - v;
      int next_r = prev_r + d_r;
      if(next_r > 2'000'000'000) error("Regret overflowing!\n" + info_str(state, prev_r, d_r, t, board, hands), debug);
      if(next_r > REGRET_FLOOR) {
        r_atom.fetch_add(d_r);
      }
      if(_log_level == SolverLogLevel::DEBUG) log_regret(actions[a_idx], d_r, next_r, debug);
    }
    return v;
  }
  else {
    auto actions = regret_node_actions(regret_storage, state, get_config().action_profile);
    int a_idx = external_sampling(actions, state, t, board, hands, indexers, eval, regret_storage, avg_storage, debug);
    PokerState next_state = state.apply(actions[a_idx]);
    return traverse_mccfr(next_state, t, i, board, hands, indexers, eval, next_regret_storage(regret_storage, a_idx, next_state, i), 
        next_avg_storage(avg_storage, a_idx, next_state, i), debug);
  }
}

template <template<typename> class StorageT>
int MCCFRSolver<StorageT>::external_sampling(const std::vector<Action>& actions, const PokerState& state, long t, const Board& board, 
    const std::vector<Hand>& hands, std::vector<CachedIndexer>& indexers, const omp::HandEvaluator& eval, StorageT<int>* regret_storage, 
    StorageT<float>* avg_storage, std::ostringstream& debug) {
  std::vector<float> freq;
  int cluster = FlatClusterMap::get_instance()->cluster(state.get_round(), indexers[state.get_active()].index(board, hands[state.get_active()], 
      state.get_round()));
  if(state.get_round() == 0 && is_preflop_frozen(t)) {
    std::atomic<float>* base_ptr = get_base_avg_ptr(avg_storage, state, cluster);
    freq = calculate_strategy(base_ptr, actions.size());
  }
  else {
    std::atomic<int>* base_ptr = get_base_regret_ptr(regret_storage, state, cluster);
    freq = calculate_strategy(base_ptr, actions.size());
  }
  int a_idx = sample_action_idx(freq);
  if(_log_level == SolverLogLevel::DEBUG) log_external_sampling(a_idx, actions, freq, state, get_config().init_state, debug);
  return a_idx;
}

template <template<typename> class StorageT>
std::string MCCFRSolver<StorageT>::track_wandb_metrics(long t) const {
  auto t_i = std::chrono::high_resolution_clock::now();  
  nlohmann::json metrics = {};
  metrics["t (M)"] = static_cast<float>(t / 1'000'000.0);
  std::ostringstream out_str;
  out_str << std::setprecision(1) << std::fixed << std::setw(7) << t / 1'000'000.0 << "M it   ";
  track_regret(metrics, out_str, t);
  track_strategy(metrics, out_str);
  auto t_f = std::chrono::high_resolution_clock::now();  
  auto dt = std::chrono::duration_cast<std::chrono::microseconds>(t_f - t_i).count();
  out_str << std::setw(8) << dt << " us (metrics)";
  Logger::dump(out_str);
  return metrics.dump();
}

template <template<typename> class StorageT>
void MCCFRSolver<StorageT>::track_strategy_by_decision(const DecisionAlgorithm& decision, nlohmann::json& metrics, bool phi) const {
  PokerState state = get_config().init_state;
  for(int p = 0; p < get_config().poker.n_players; ++p) {
    if(state.active_players() <= 1) return;
    auto actions = available_actions(state, get_config().action_profile);
    PokerRange base_range = get_config().init_ranges[state.get_active()];
    base_range.remove_cards(get_config().init_board);
    for(Action a : actions) {
      PokerRange action_range = build_action_range(base_range, a, state, get_config().init_board, decision);
      double freq = (base_range * action_range).n_combos() / base_range.n_combos();
      std::string data_label = pos_to_str(state.get_active(), get_config().poker.n_players) + " " + a.to_string() + (!phi ? " (regrets)" : " (phi)");
      metrics[data_label] = freq;
    }
    state = state.apply(std::find(actions.begin(), actions.end(), Action::FOLD) != actions.end() ? Action::FOLD : Action::CHECK_CALL);
  }
}

// to allow use of MCCFRSolver::traverse_mccfr friend in benchmark_mccfr.cpp without moving MCCFRSolver::traverse_mccfr to the header mccfr.hpp 
// (because it's a template and used in a different translation unit)
template class pluribus::MCCFRSolver<pluribus::StrategyStorage>; 

// ==========================================================================================
// || MappedSolver
// ==========================================================================================

float MappedSolver::frequency(Action action, const PokerState& state, const Board& board, const Hand& hand) const {
  StrategyDecision decision{get_strategy(), get_config().action_profile};
  return decision.frequency(action, state, board, hand);
}

std::atomic<int>* MappedSolver::get_base_regret_ptr(StrategyStorage<int>* storage, const PokerState& state, int cluster) {
  return &storage->get(state, cluster);
}

std::vector<Action> MappedSolver::regret_node_actions(StrategyStorage<int>* storage, const PokerState& state, const ActionProfile& profile) const {
  return available_actions(state, profile);
}

std::vector<Action> MappedSolver::avg_node_actions(StrategyStorage<float>* storage, const PokerState& state, const ActionProfile& profile) const {
  return available_actions(state, profile);
}

void MappedSolver::track_strategy(nlohmann::json& metrics, std::ostringstream& out_str) const {
  track_strategy_by_decision(StrategyDecision{get_strategy(), get_config().action_profile}, metrics, false);
}

// ==========================================================================================
// || TreeSolver
// ==========================================================================================

float TreeSolver::frequency(Action action, const PokerState& state, const Board& board, const Hand& hand) const {
  TreeDecision<int> decision{_regrets_root.get(), get_config().init_state};
  return decision.frequency(action, state, board, hand);
}

std::atomic<int>* TreeSolver::get_base_regret_ptr(TreeStorageNode<int>* storage, const PokerState& state, int cluster) { 
  return storage->get(cluster); 
};

TreeStorageNode<int>* TreeSolver::init_regret_storage() { 
  if(!_regrets_root) {
    _regrets_tree_config = make_tree_config();
    _regrets_root = std::make_unique<TreeStorageNode<int>>(get_config().init_state, _regrets_tree_config);
  }
  return _regrets_root.get(); 
};

TreeStorageNode<int>* TreeSolver::next_regret_storage(TreeStorageNode<int>* storage, int action_idx, const PokerState& next_state, int i) {
  return !is_terminal(next_state, i) ? storage->apply_index(action_idx, next_state) : nullptr;
}

std::vector<Action> TreeSolver::regret_node_actions(TreeStorageNode<int>* storage, const PokerState& state, const ActionProfile& profile) const {
  return storage->get_actions();
}

std::vector<Action> TreeSolver::avg_node_actions(TreeStorageNode<float>* storage, const PokerState& state, const ActionProfile& profile) const {
  return storage->get_actions();
}


void TreeSolver::track_strategy(nlohmann::json& metrics, std::ostringstream& out_str) const {
  track_strategy_by_decision(TreeDecision<int>{_regrets_root.get(), get_config().init_state}, metrics, false);
}

// ==========================================================================================
// || BlueprintSolver
// ==========================================================================================

template <template<typename> class StorageT>
bool BlueprintSolver<StorageT>::is_update_terminal(const PokerState& state, int i) const {
  return state.get_winner() != -1 || state.get_round() > 0 || state.get_players()[i].has_folded();
}

template <template<typename> class StorageT>
void BlueprintSolver<StorageT>::update_strategy(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, 
    StorageT<int>* regret_storage, StorageT<float>* avg_storage, std::ostringstream& debug) {
  if(is_update_terminal(state, i)) {
    return;
  }
  else if(state.get_active() == i) {
    auto actions = this->avg_node_actions(avg_storage, state, this->get_config().action_profile);
    int cluster = FlatClusterMap::get_instance()->cluster(state.get_round(), board, hands[i]);
    std::atomic<int>* base_ptr = this->get_base_regret_ptr(regret_storage, state, cluster);
    auto freq = calculate_strategy(base_ptr, actions.size());
    int a_idx = sample_action_idx(freq);
    if(this->get_log_level() != SolverLogLevel::NONE) {
      debug << "Update strategy: " << relative_history_str(state, this->get_config().init_state) << "\n";
      debug << "\t" << hands[i].to_string() << ": (cluster=" << cluster << ")\n\t";
      for(int ai = 0; ai < actions.size(); ++ai) {
        debug << actions[ai].to_string() << "=" << std::setprecision(2) << std::fixed << freq[ai] << "  ";
      }
      debug << "\n";
    }
    this->get_base_avg_ptr(avg_storage, state, cluster)[a_idx].fetch_add(1.0f);
    PokerState next_state = state.apply(actions[a_idx]);
    update_strategy(next_state, i, board, hands, this->next_regret_storage(regret_storage, a_idx, next_state, i), 
        this->next_avg_storage(avg_storage, a_idx, next_state, i), debug);
  }
  else {
    auto actions = this->avg_node_actions(avg_storage, state, this->get_config().action_profile);
    for(int a_idx = 0; a_idx < actions.size(); ++a_idx) {
      PokerState next_state = state.apply(actions[a_idx]);
      update_strategy(next_state, i, board, hands, this->next_regret_storage(regret_storage, a_idx, next_state, i), 
          this->next_avg_storage(avg_storage, a_idx, next_state, i), debug);
    }
  }
}

template <template<typename> class StorageT>
void BlueprintSolver<StorageT>::on_step(long t,int i, const std::vector<Hand>& hands, std::ostringstream& debug) {
  if(t > 0 && t % get_blueprint_config().strategy_interval == 0 && t < get_blueprint_config().preflop_threshold) {
    if(this->get_log_level() != SolverLogLevel::NONE) debug << "============== Updating strategy ==============\n";
    update_strategy(this->get_config().init_state, i, this->get_config().init_board, hands, 
        this->init_regret_storage(), this->init_avg_storage(), debug);
  }
}

template <template<typename> class StorageT>
bool BlueprintSolver<StorageT>::should_prune(long t) const {
  if(t < _bp_config.prune_thresh) return false;
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  return dist(GlobalRNG::instance()) > 0.95;
}

template <template<typename> class StorageT>
long BlueprintSolver<StorageT>::next_step(long t, long T) const {
  return std::min(std::min(_bp_config.next_discount_step(t, T), _bp_config.next_snapshot_step(t, T)), T);
}

// ==========================================================================================
// || RealTimeSolver
// ==========================================================================================

template <template<typename> class StorageT>
int RealTimeSolver<StorageT>::terminal_utility(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, int stack_size, 
    std::vector<CachedIndexer>& indexers, const omp::HandEvaluator& eval) const {
  if(state.has_biases() && state.get_active() != state._first_bias) {
    Logger::error("Active player changed after biasing. Active=" + std::to_string(static_cast<int>(state.get_active())) + 
        ", First bias=" + std::to_string(static_cast<int>(state._first_bias)));
  }
  PokerState curr_state = state;
  while(!state.is_terminal() && !state.get_players()[i].has_folded()) {
    curr_state = curr_state.apply(_action_provider.next_action(indexers[state.get_active()], state, hands, board, _bp.get()));
  }
  return utility(curr_state, i, board, hands, stack_size, eval);
}

template <template<typename> class StorageT>
std::vector<Action> RealTimeSolver<StorageT>::available_actions(const PokerState& state, const ActionProfile& profile) const {
  if(state.get_round() >= _rt_config.terminal_round || state.get_bet_level() >= _rt_config.terminal_bet_level) {
    return _rt_config.bias_profile.get_actions(0, 0, 0, 0);
  }
  else {
    return valid_actions(state, profile);
  }
}

// ==========================================================================================
// || MappedBlueprintSolver
// ==========================================================================================

MappedBlueprintSolver::MappedBlueprintSolver(const SolverConfig& config, const BlueprintSolverConfig& bp_config) 
    : MappedSolver{config, 200}, BlueprintSolver{config, bp_config}, MCCFRSolver{config}, _phi{config.action_profile, 169} {}

std::atomic<float>* MappedBlueprintSolver::get_base_avg_ptr(StrategyStorage<float>* storage, const PokerState& state, int cluster) {
  return &storage->get(state, cluster);
}

bool MappedBlueprintSolver::operator==(const MappedBlueprintSolver& other) const {
  return MappedSolver::operator==(other) && BlueprintSolver::operator==(other) && _phi == other._phi;
}

void MappedBlueprintSolver::track_strategy(nlohmann::json& metrics, std::ostringstream& out_str) const {
  MappedSolver::track_strategy(metrics, out_str);
  track_strategy_by_decision(StrategyDecision{get_phi(), get_config().action_profile}, metrics, true);
}

void MappedBlueprintSolver::track_regret(nlohmann::json& metrics, std::ostringstream& out_str, long t) const {
  long avg_regret = 0;
  for(auto& r : get_strategy().data()) avg_regret += std::max(r.load(), 0); // should be sum of the maximum regret at each infoset, not sum of all regrets
  avg_regret /= t;
  out_str << std::setw(8) << avg_regret << " avg regret   ";
  metrics["avg_regret"] = static_cast<int>(avg_regret);
}


// ==========================================================================================
// || MappedRealTimeSolver
// ==========================================================================================

MappedRealTimeSolver::MappedRealTimeSolver(const std::shared_ptr<const SampledBlueprint> bp, const RealTimeSolverConfig& rt_config)
    : MappedSolver{bp->get_config(), MAX_COMBOS}, RealTimeSolver{bp, rt_config}, MCCFRSolver{bp->get_config()} {
  if(rt_config.terminal_round < 1 || rt_config.terminal_round > 4) 
    Logger::error("Invalid terminal round: " + std::to_string(rt_config.terminal_round));
  if(rt_config.terminal_bet_level < 1) Logger::error("Invalid terminal bet level: " + std::to_string(rt_config.terminal_bet_level));
  if(rt_config.discount_interval < 1) Logger::error("Invalid discount interval: " + std::to_string(rt_config.discount_interval));
  if(rt_config.log_interval < 1) Logger::error("Invalid log interval: " + std::to_string(rt_config.log_interval));
}

void MappedRealTimeSolver::track_regret(nlohmann::json& metrics, std::ostringstream& out_str, long t) const {
  long max_regret_sum = 0;
  for(const auto& entry : get_strategy().history_map()) {
    PokerState state{get_config().poker};
    state = state.apply(entry.first);
    auto actions = available_actions(state, get_config().action_profile);
    for(int c = 0; c < get_strategy().n_clusters(); ++c) {
      int max_regret = 0;
      size_t base_idx = entry.second.idx + c * actions.size();
      for(int a_idx = 0; a_idx < actions.size(); ++a_idx) {
        max_regret = std::max(get_strategy()[base_idx + a_idx].load(), max_regret);
      }
      max_regret_sum += max_regret;
    }
  }
  double avg_max_regrets = max_regret_sum / static_cast<double>(get_iteration());
  out_str << std::setw(8) << std::fixed << std::setprecision(2) << avg_max_regrets << " avg max regret   ";
  metrics["avg_regret"] = static_cast<int>(avg_max_regrets);
}

// ==========================================================================================
// || TreeBlueprintSolver
// ==========================================================================================

const std::shared_ptr<const TreeStorageConfig> TreeBlueprintSolver::make_tree_config() const {
  return std::make_shared<TreeStorageConfig>(TreeStorageConfig{
    [](const PokerState& state) { return state.get_round() == 0 ? 169 : 200; },
    [this](const PokerState& state) { return available_actions(state, this->get_config().action_profile); }
  });
}

TreeBlueprintSolver::TreeBlueprintSolver(const SolverConfig& config, const BlueprintSolverConfig& bp_config) 
    : TreeSolver{config}, BlueprintSolver{config, bp_config}, MCCFRSolver{config} {}

std::atomic<float>* TreeBlueprintSolver::get_base_avg_ptr(TreeStorageNode<float>* storage, const PokerState& state, int cluster) {
  return storage->get(cluster);
}

TreeStorageNode<float>* TreeBlueprintSolver::init_avg_storage() { 
  if(!_phi_root) {
    _phi_tree_config = make_tree_config();
    _phi_root = std::make_unique<TreeStorageNode<float>>(get_config().init_state, _phi_tree_config);
  }
  return _phi_root.get(); 
};

TreeStorageNode<float>* TreeBlueprintSolver::next_avg_storage(TreeStorageNode<float>* storage, int action_idx, const PokerState& next_state, int i) {
  return !is_update_terminal(next_state, i) ? storage->apply_index(action_idx, next_state) : nullptr;
}

void TreeBlueprintSolver::track_strategy(nlohmann::json& metrics, std::ostringstream& out_str) const {
  TreeSolver::track_strategy(metrics, out_str);
  track_strategy_by_decision(TreeDecision<float>{_phi_root.get(), get_config().init_state}, metrics, true);
}

long sum_node_regrets(const TreeStorageNode<int>* node) {
  long r = 0L;
  for(int c = 0; c < node->get_n_clusters(); ++c) {
    int max_r = 0L;
    for(int a_idx = 0; a_idx < node->get_actions().size(); ++a_idx) {
      max_r = std::max(node->get(c, a_idx)->load(), max_r);
    }
    r += max_r;
  }
  for(int a_idx = 0; a_idx < node->get_actions().size(); ++a_idx) {
    if(node->is_allocated(a_idx)) r += sum_node_regrets(node->apply_index(a_idx));
  }
  return r;
}

void TreeBlueprintSolver::track_regret(nlohmann::json& metrics, std::ostringstream& out_str, long t) const {
  long avg_regret = sum_node_regrets(get_regrets_root()) / t; // should be sum of the maximum regret at each infoset, not sum of all regrets
  out_str << std::setw(8) << avg_regret << " avg regret   ";
  metrics["avg max regret"] = static_cast<int>(avg_regret);
}

}
