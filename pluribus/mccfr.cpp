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
        write_to_file(_metrics_dir / metrics_fn.str(), build_wandb_metrics(t));
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

void log_net_ev(int ev, float ev_exact, const PokerState& state, const PokerState& init_state, std::ostringstream& debug) {
  debug << "Net EV: " << relative_history_str(state, init_state) << "\n";
  debug << "\tu(sigma) = " << std::setprecision(2) << std::fixed << ev << " (exact=" << ev_exact << ")\n";
  if(state.get_round() == 0) debug << "Preflop frozen, skipping regret update.\n";
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
  if(is_terminal(state) || state.get_players()[i].has_folded()) {
    int u = terminal_utility(state, i, board, hands, get_config().poker.n_chips, indexers, eval);
    if(_log_level == SolverLogLevel::DEBUG) log_utility(u, state, get_config().init_state, hands, debug);
    return u;
  }
  else if(state.get_active() == i) {
    auto actions = available_actions(state, get_config().action_profile);
    int cluster = FlatClusterMap::get_instance()->cluster(state.get_round(), indexers[state.get_active()].index(board, hands[i], state.get_round()));
    if(_log_level == SolverLogLevel::DEBUG) debug << "Cluster:" << cluster << "\n";
    std::atomic<int>* base_ptr = get_base_regret_ptr(regret_storage, state, cluster);
    auto freq = calculate_strategy(base_ptr, actions.size());

    std::unordered_map<Action, int> values;
    float v_exact = 0;
    for(int a_idx = 0; a_idx < actions.size(); ++a_idx) {
      Action a = actions[a_idx];
      if(base_ptr[a_idx].load() > PRUNE_CUTOFF) {
        int v_a = traverse_mccfr_p(state.apply(a), t, i, board, hands, indexers, eval, regret_storage, avg_storage, debug);
        values[a] = v_a;
        v_exact += freq[a_idx] * v_a;
        if(_log_level == SolverLogLevel::DEBUG) log_action_ev(a, freq[a_idx], v_a, state, get_config().init_state, debug);
      }
    }
    int v = round(v_exact);
    if(_log_level == SolverLogLevel::DEBUG) log_net_ev(v, v_exact, state, get_config().init_state, debug);
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
    Action a = external_sampling(state, t, board, hands, indexers, eval, regret_storage, avg_storage, debug);
    return traverse_mccfr_p(state.apply(a), t, i, board, hands, indexers, eval, regret_storage, avg_storage, debug);
  }
}

template <template<typename> class StorageT>
int MCCFRSolver<StorageT>::traverse_mccfr(const PokerState& state, long t, int i, const Board& board, const std::vector<Hand>& hands, 
    std::vector<CachedIndexer>& indexers, const omp::HandEvaluator& eval, StorageT<int>* regret_storage, StorageT<float>* avg_storage, 
    std::ostringstream& debug) {
  if(is_terminal(state) || state.get_players()[i].has_folded()) {
    int u = terminal_utility(state, i, board, hands, get_config().poker.n_chips, indexers, eval);
    if(_log_level == SolverLogLevel::DEBUG) log_utility(u, state, get_config().init_state, hands, debug);
    return u;
  }
  else if(state.get_active() == i) {
    auto actions = available_actions(state, get_config().action_profile);
    int cluster = FlatClusterMap::get_instance()->cluster(state.get_round(), indexers[state.get_active()].index(board, hands[i], state.get_round()));
    if(_log_level == SolverLogLevel::DEBUG) debug << "Cluster:" << cluster << "\n";
    std::atomic<int>* base_ptr = get_base_regret_ptr(regret_storage, state, cluster);
    auto freq = calculate_strategy(base_ptr, actions.size());
    std::unordered_map<Action, int> values;
    float v_exact = 0;
    for(int a_idx = 0; a_idx < actions.size(); ++a_idx) {
      Action a = actions[a_idx];
      int v_a = traverse_mccfr(state.apply(a), t, i, board, hands, indexers, eval, regret_storage, avg_storage, debug);
      values[a] = v_a;
      v_exact += freq[a_idx] * v_a;
      if(_log_level == SolverLogLevel::DEBUG) log_action_ev(a, freq[a_idx], v_a, state, get_config().init_state, debug);
    }
    int v = round(v_exact);
    if(_log_level == SolverLogLevel::DEBUG) log_net_ev(v, v_exact, state, get_config().init_state, debug);
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
    Action a = external_sampling(state, t, board, hands, indexers, eval, regret_storage, avg_storage, debug);
    return traverse_mccfr_p(state.apply(a), t, i, board, hands, indexers, eval, regret_storage, avg_storage, debug);
  }
}

template <template<typename> class StorageT>
Action MCCFRSolver<StorageT>::external_sampling(const PokerState& state, long t, const Board& board, const std::vector<Hand>& hands, 
    std::vector<CachedIndexer>& indexers, const omp::HandEvaluator& eval, StorageT<int>* regret_storage, StorageT<float>* avg_storage, 
    std::ostringstream& debug) {
  auto actions = available_actions(state, get_config().action_profile);
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
  Action a = actions[sample_action_idx(freq)];
  if(_log_level == SolverLogLevel::DEBUG) log_external_sampling(a, actions, freq, state, get_config().init_state, debug);
  return a;
}

// to allow use of MCCFRSolver::traverse_mccfr friend in benchmark_mccfr.cpp without moving MCCFRSolver::traverse_mccfr to the header mccfr.hpp 
// (because it's a template and used in a different translation unit)
template class pluribus::MCCFRSolver<pluribus::StrategyStorage>; 

// ==========================================================================================
// || MappedSolver
// ==========================================================================================

std::atomic<int>* MappedSolver::get_base_regret_ptr(StrategyStorage<int>* storage, const PokerState& state, int cluster) {
  return &storage->get(state, cluster);
}

template <class T>
void MappedSolver::log_strategy(const StrategyStorage<T>& strat, const SolverConfig& config, nlohmann::json& metrics, bool phi) const {
  PokerState state = config.init_state;
  for(int p = 0; p < config.poker.n_players; ++p) {
    if(state.active_players() <= 1) return;
    auto actions = available_actions(state, config.action_profile);
    PokerRange range_copy = config.init_ranges[state.get_active()];
    
    auto ranges = build_renderable_ranges(strat, config.action_profile, state, config.init_board, range_copy);
    for(Action a : actions) {
      double freq = ranges.at(a).get_range().n_combos() / MAX_COMBOS;
      std::string data_label = pos_to_str(state.get_active(), config.poker.n_players) + " " + a.to_string() + (!phi ? " (regrets)" : " (phi)");
      metrics[data_label] = freq;
    }
    state = state.apply(std::find(actions.begin(), actions.end(), Action::FOLD) != actions.end() ? Action::FOLD : Action::CHECK_CALL);
  }
}

float MappedSolver::frequency(Action action, const PokerState& state, const Board& board, const Hand& hand) const {
    StrategyDecision decision{get_strategy(), get_config().action_profile};
    return decision.frequency(action, state, board, hand);
  }

// ==========================================================================================
// || BlueprintSolver
// ==========================================================================================

template <template<typename> class StorageT>
void BlueprintSolver<StorageT>::update_strategy(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, 
    StorageT<int>* regret_storage, StorageT<float>* avg_storage, std::ostringstream& debug) {
  if(state.get_winner() != -1 || state.get_round() > 0 || state.get_players()[i].has_folded()) {
    return;
  }
  else if(state.get_active() == i) {
    auto actions = this->available_actions(state, this->get_config().action_profile);
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
    update_strategy(state.apply(actions[a_idx]), i, board, hands, regret_storage, avg_storage, debug);
  }
  else {
    for(Action action : this->available_actions(state, this->get_config().action_profile)) {
      update_strategy(state.apply(action), i, board, hands, regret_storage, avg_storage, debug);
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
  if(state.get_active() != state._first_bias) {
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

std::string MappedBlueprintSolver::build_wandb_metrics(long t) const {
  long avg_regret = 0;
  for(auto& r : get_strategy().data()) avg_regret += std::max(r.load(), 0); // should be sum of the maximum regret at each infoset, not sum of all regrets
  avg_regret /= t;
  std::ostringstream buf;
  buf << std::setprecision(1) << std::fixed << std::setw(7) << t / 1'000'000.0 << "M it   " 
      << std::setw(12) << avg_regret << " avg regret";
  Logger::dump(buf);
  nlohmann::json metrics = {
    {"avg_regret", static_cast<int>(avg_regret)},
    {"t (M)", static_cast<float>(t / 1'000'000.0)}
  };
  log_strategy(get_strategy(), get_config(), metrics, false);
  log_strategy(get_phi(), get_config(), metrics, true);
  return metrics.dump();
}

bool MappedBlueprintSolver::operator==(const MappedBlueprintSolver& other) const {
  return MappedSolver::operator==(other) && BlueprintSolver::operator==(other) && _phi == other._phi;
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

std::string MappedRealTimeSolver::build_wandb_metrics(long t) const {
  auto t_i = std::chrono::high_resolution_clock::now();  
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
  double avg_max_regrets = max_regret_sum / static_cast<double>(t);
  std::ostringstream buf;
  buf << std::setprecision(1) << std::fixed << std::setw(7) << t / 1'000'000.0 << "M it   " 
      << std::setw(8) << avg_max_regrets << " avg max regret   ";
  nlohmann::json metrics = {
    {"avg_regret", static_cast<int>(avg_max_regrets)},
    {"t (M)", static_cast<float>(t / 1'000'000.0)}
  };
  log_strategy(get_strategy(), get_config(), metrics, false);
  auto t_f = std::chrono::high_resolution_clock::now();  
  auto dt = std::chrono::duration_cast<std::chrono::microseconds>(t_f - t_i).count();
  buf << std::setw(8) << dt << " us (metrics)";
  Logger::dump(buf);
  return metrics.dump();
}

}
