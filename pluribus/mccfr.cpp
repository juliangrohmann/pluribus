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
#include <pluribus/logging.hpp>
#include <pluribus/mccfr.hpp>

namespace pluribus {

static const int PRUNE_CUTOFF = -300'000'000;
static const int REGRET_FLOOR = -310'000'000;

MCCFRConfig::MCCFRConfig(const PokerConfig& poker_) 
    : poker{poker_}, action_profile{BlueprintActionProfile{poker_.n_players}}, init_state{poker_} {
  for(int i = 0; i < poker_.n_players; ++i) init_ranges.push_back(PokerRange::full());
}

MCCFRConfig::MCCFRConfig(int n_players, int n_chips, int ante) 
    : MCCFRConfig{PokerConfig{n_players, n_chips, ante}} {}

std::string MCCFRConfig::to_string() const {
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

void BlueprintTrainerConfig::set_iterations(const BlueprintTimingConfig& timings, long it_per_min) {
  preflop_threshold = timings.preflop_threshold_m * it_per_min;
  snapshot_interval = timings.snapshot_interval_m * it_per_min;
  prune_thresh = timings.prune_thresh_m * it_per_min;
  lcfr_thresh = timings.lcfr_thresh_m * it_per_min;
  discount_interval = timings.discount_interval_m * it_per_min;
  log_interval = timings.log_interval_m * it_per_min;
}

long BlueprintTrainerConfig::next_discount_step(long t, long T) const {
  long next_disc = ((t + 1) / discount_interval + 1) * discount_interval;
  return next_disc < lcfr_thresh ? next_disc : T + 1;
}

long BlueprintTrainerConfig::next_snapshot_step(long t, long T) const {
    long next_snap = std::max((t - preflop_threshold + 1) / snapshot_interval + 1, 0L) * snapshot_interval + preflop_threshold;
    return next_snap < T ? next_snap : T;
}

bool BlueprintTrainerConfig::is_discount_step(long t) const {
  return t < lcfr_thresh && t % discount_interval == 0;
}

bool BlueprintTrainerConfig::is_snapshot_step(long t, long T) const {
  return t == T || (t - preflop_threshold) % snapshot_interval == 0;
}

int sample_action_idx(const std::vector<float>& freq) {
  std::discrete_distribution<> dist(freq.begin(), freq.end());
  return dist(GlobalRNG::instance());
}

int utility(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, int stack_size, const omp::HandEvaluator& eval) {
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

BlueprintTrainerConfig::BlueprintTrainerConfig() {
  set_iterations(BlueprintTimingConfig{}, 10'000'000);
}

std::string BlueprintTrainerConfig::to_string() const {
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

MCCFRTrainer::MCCFRTrainer(const MCCFRConfig& mccfr_config) : _mccfr_config{mccfr_config} {
  if(mccfr_config.init_state.get_players().size() != mccfr_config.poker.n_players) Logger::error("Player number mismatch");
}

MCCFRTrainer::~MCCFRTrainer() = default;

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

void MCCFRTrainer::mccfr_p(long t_plus) {
  if(!create_dir(_snapshot_dir)) Logger::error("Failed to create snapshot dir: " + _snapshot_dir.string());
  if(!create_dir(_metrics_dir)) Logger::error("Failed to create metrics dir: " + _metrics_dir.string());
  if(!create_dir(_log_dir)) Logger::error("Failed to create log dir: " + _log_dir.string());

  long T = _t + t_plus;
  Logger::log("MCCFRTrainer --- Initializing HandIndexer...");
  Logger::log(HandIndexer::get_instance() ? "Success." : "Failure.");
  Logger::log("MCCFRTrainer --- Initializing FlatClusterMap...");
  Logger::log(FlatClusterMap::get_instance() ? "Success." : "Failure.");
  Logger::log(_mccfr_config.to_string());
  on_start();

  bool full_ranges = are_full_ranges(_mccfr_config.init_ranges);
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
      thread_local Deck deck{_mccfr_config.init_board};
      thread_local Board board;
      thread_local std::vector<Hand> hands{static_cast<size_t>(_mccfr_config.poker.n_players)};
      thread_local std::ostringstream debug;
      if(_log_level == BlueprintLogLevel::DEBUG) debug << "============== t = " << t << " ==============\n";
      // if(t % (_config.log_interval) == 0) log_metrics(t);
      if(should_log(t)) {
        std::ostringstream metrics_fn;
        metrics_fn << std::setprecision(1) << std::fixed << t / 1'000'000.0 << ".json";
        write_to_file(_metrics_dir / metrics_fn.str(), build_wandb_metrics(t));
      }
      for(int i = 0; i < _mccfr_config.poker.n_players; ++i) {
        if(_log_level == BlueprintLogLevel::DEBUG) debug << "============== i = " << i << " ==============\n";
        deck.shuffle();
        board.deal(deck, _mccfr_config.init_board);
        if(full_ranges) {
          for(auto& hand : hands) hand.deal(deck);
        }
        else {
          std::unordered_set<uint8_t> dead_cards;
          std::copy(board.cards().begin(), board.cards().end(), std::inserter(dead_cards, dead_cards.end()));
          for(int p_idx = 0; p_idx < _mccfr_config.poker.n_players; ++p_idx) {
            Logger::error("Biased MCCFR sampling not implemented.");
            // hands[p_idx] = _config.init_ranges[p_idx].sample(dead_cards);
            dead_cards.insert(hands[p_idx].cards()[0]);
            dead_cards.insert(hands[p_idx].cards()[1]);
          }
        }

        on_step(t, i, hands, debug);
        // if(t > _config.prune_thresh) {
        if(should_prune(t)) {
          if(_log_level == BlueprintLogLevel::DEBUG) debug << "============== Traverse MCCFR-P ==============\n";
          traverse_mccfr_p(_mccfr_config.init_state, t, i, board, hands, eval, debug);
        }
        else {
          if(_log_level == BlueprintLogLevel::DEBUG) debug << "============== Traverse MCCFR ==============\n";
          traverse_mccfr(_mccfr_config.init_state, t, i, board, hands, eval, debug);
        }
      }
      if(_log_level == BlueprintLogLevel::DEBUG) {
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
      lcfr_discount(get_regrets(), d);
      lcfr_discount(get_avg_strategy(), d);
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

void MCCFRTrainer::error(const std::string& msg, const std::ostringstream& debug) const {
  std::string error_msg = msg;
  if(_log_level == BlueprintLogLevel::DEBUG) {
    auto debug_dir = _log_dir / ("thread" + std::to_string(omp_get_thread_num()) + ".error");
    write_to_file(debug_dir, debug.str() + "\nRUNTIME ERROR: " + msg);
    error_msg += "\nDebug logs written to " + debug_dir.string();
  }
  Logger::error(error_msg);
}

int MCCFRTrainer::traverse_mccfr_p(const PokerState& state, long t, int i, const Board& board, const std::vector<Hand>& hands, 
                                       const omp::HandEvaluator& eval, std::ostringstream& debug) {
  if(state.is_terminal() || state.get_players()[i].has_folded()) {
    int u = utility(state, i, board, hands, _mccfr_config.poker.n_chips, eval);
    if(_log_level == BlueprintLogLevel::DEBUG) log_utility(u, state, hands, debug);
    return u;
  }
  else if(state.get_active() == i) {
    auto actions = valid_actions(state, _mccfr_config.action_profile);
    int cluster = FlatClusterMap::get_instance()->cluster(state.get_round(), board, hands[i]);
    if(_log_level == BlueprintLogLevel::DEBUG) debug << "Cluster:" << cluster << "\n";
    size_t base_idx = get_regrets()->index(state, cluster);
    auto freq = calculate_strategy(*get_regrets(), base_idx, actions.size());

    std::unordered_map<Action, int> values;
    float v_exact = 0;
    for(int a_idx = 0; a_idx < actions.size(); ++a_idx) {
      Action a = actions[a_idx];
      if((*get_regrets())[base_idx + a_idx].load() > PRUNE_CUTOFF) {
        int v_a = traverse_mccfr_p(state.apply(a), t, i, board, hands, eval, debug);
        values[a] = v_a;
        v_exact += freq[a_idx] * v_a;
        if(_log_level == BlueprintLogLevel::DEBUG) log_action_ev(a, freq[a_idx], v_a, state, debug);
      }
    }
    int v = round(v_exact);
    if(_log_level == BlueprintLogLevel::DEBUG) log_net_ev(v, v_exact, state, debug);
    if(state.get_round() == 0 && is_preflop_frozen(t)) return v;
    for(int a_idx = 0; a_idx < actions.size(); ++a_idx) {
      Action a = actions[a_idx];
      if(values.find(a) != values.end()) {
        int prev_r = (*get_regrets())[base_idx + a_idx].load();
        int d_r = values[a] - v;
        int next_r = prev_r + d_r;
        if(next_r > 2'000'000'000) error("Regret overflowing!\n" + info_str(state, prev_r, d_r, t, board, hands), debug);
        int total_r = std::max(next_r, REGRET_FLOOR);
        (*get_regrets())[base_idx + a_idx].store(total_r);
        if(_log_level == BlueprintLogLevel::DEBUG) log_regret(actions[a_idx], d_r, total_r, debug);
      }
    }
    return v;
  }
  else {
    auto actions = valid_actions(state, _mccfr_config.action_profile);
    std::vector<float> freq;
    if(state.get_round() == 0 && t > is_preflop_frozen(t)) {
      freq = state_to_freq(state, board, hands[state.get_active()], actions.size(), *get_avg_strategy());
    }
    else {
      freq = state_to_freq(state, board, hands[state.get_active()], actions.size(), *get_regrets());
    }
    Action a = actions[sample_action_idx(freq)];
    return traverse_mccfr_p(state.apply(a), t, i, board, hands, eval, debug);
  }
}

std::string relative_history_str(const PokerState& state, const PokerState& init_state) {
  return state.get_action_history().slice(init_state.get_action_history().size()).to_string();
}

int MCCFRTrainer::traverse_mccfr(const PokerState& state, long t, int i, const Board& board, const std::vector<Hand>& hands, 
                                     const omp::HandEvaluator& eval, std::ostringstream& debug) {
  if(state.is_terminal() || state.get_players()[i].has_folded()) {
    int u = utility(state, i, board, hands, _mccfr_config.poker.n_chips, eval);
    if(_log_level == BlueprintLogLevel::DEBUG) log_utility(u, state, hands, debug);
    return u;
  }
  else if(state.get_active() == i) {
    auto actions = valid_actions(state, _mccfr_config.action_profile);
    int cluster = FlatClusterMap::get_instance()->cluster(state.get_round(), board, hands[i]);
    if(_log_level == BlueprintLogLevel::DEBUG) debug << "Cluster:" << cluster << "\n";
    size_t base_idx = get_regrets()->index(state, cluster);
    auto freq = calculate_strategy(*get_regrets(), base_idx, actions.size());
    std::unordered_map<Action, int> values;
    float v_exact = 0;
    for(int a_idx = 0; a_idx < actions.size(); ++a_idx) {
      Action a = actions[a_idx];
      int v_a = traverse_mccfr(state.apply(a), t, i, board, hands, eval, debug);
      values[a] = v_a;
      v_exact += freq[a_idx] * v_a;
      if(_log_level == BlueprintLogLevel::DEBUG) log_action_ev(a, freq[a_idx], v_a, state, debug);
    }
    int v = round(v_exact);
    if(_log_level == BlueprintLogLevel::DEBUG) log_net_ev(v, v_exact, state, debug);
    if(state.get_round() == 0 && t > is_preflop_frozen(t)) return v;

    for(int a_idx = 0; a_idx < actions.size(); ++a_idx) {
      int prev_r = (*get_regrets())[base_idx + a_idx].load();
      int d_r = values[actions[a_idx]] - v;
      int next_r = prev_r + d_r;
      if(next_r > 2'000'000'000) error("Regret overflowing!\n" + info_str(state, prev_r, d_r, t, board, hands), debug);
      int total_r = std::max(next_r, REGRET_FLOOR);
      (*get_regrets())[base_idx + a_idx].store(total_r);
      if(_log_level == BlueprintLogLevel::DEBUG) log_regret(actions[a_idx], d_r, total_r, debug);
    }
    return v;
  }
  else {
    auto actions = valid_actions(state, _mccfr_config.action_profile);
    std::vector<float> freq;
    if(state.get_round() == 0 && t > is_preflop_frozen(t)) {
      freq = state_to_freq(state, board, hands[state.get_active()], actions.size(), *get_avg_strategy());
    }
    else {
      freq = state_to_freq(state, board, hands[state.get_active()], actions.size(), *get_regrets());
    }
    Action a = actions[sample_action_idx(freq)];
    if(_log_level == BlueprintLogLevel::DEBUG) log_external_sampling(a, actions, freq, state, debug);
    return traverse_mccfr(state.apply(a), t, i, board, hands, eval, debug);
  }
}

void MCCFRTrainer::log_utility(int utility, const PokerState& state, const std::vector<Hand>& hands, std::ostringstream& debug) const {
  debug << "Terminal: " << relative_history_str(state, _mccfr_config.init_state) << "\n";
  debug << "\tHands: ";
  for(int p_idx = 0; p_idx < state.get_players().size(); ++p_idx) {
    debug << hands[p_idx].to_string() << " ";
  }
  debug << "\n";
  debug << "\tu(z) = " << utility << "\n";
}

void MCCFRTrainer::log_action_ev(Action a, float freq, int ev, const PokerState& state, std::ostringstream& debug) const {
  debug << "Action EV: " << relative_history_str(state, _mccfr_config.init_state) << "\n";
  debug << "\tu(" << a.to_string() << ") @ " << std::setprecision(2) << std::fixed << freq << " = " << ev << "\n";
}

void MCCFRTrainer::log_net_ev(int ev, float ev_exact, const PokerState& state, std::ostringstream& debug) const {
  debug << "Net EV: " << relative_history_str(state, _mccfr_config.init_state) << "\n";
  debug << "\tu(sigma) = " << std::setprecision(2) << std::fixed << ev << " (exact=" << ev_exact << ")\n";
  if(state.get_round() == 0) debug << "Preflop frozen, skipping regret update.\n";
}

void MCCFRTrainer::log_regret(Action a, int d_r, int total_r, std::ostringstream& debug) const {
  debug << "\tR(" << a.to_string() << ") = " << d_r << "\n";
  debug << "\tcum R(" << a.to_string() << ") = " << total_r << "\n";
}

void MCCFRTrainer::log_external_sampling(Action sampled, const std::vector<Action>& actions, const std::vector<float>& freq, 
                                             const PokerState& state, std::ostringstream& debug) const {
  debug << "Sampling: " << relative_history_str(state, _mccfr_config.init_state) << "\n\t";
  for(int a_idx = 0; a_idx < actions.size(); ++a_idx) {
    debug << std::setprecision(2) << std::fixed << actions[a_idx].to_string() << "=" << freq[a_idx] << " ";
  }
  debug << "\n";
  debug << "\tSampled: " << sampled.to_string() << "\n";
}

BlueprintTrainer::BlueprintTrainer(const BlueprintTrainerConfig& bp_config, const MCCFRConfig& mccfr_config) 
    : MCCFRTrainer{mccfr_config}, _regrets{mccfr_config.action_profile, 200}, _phi{mccfr_config.action_profile, 169}, _bp_config{bp_config} {}

void BlueprintTrainer::update_strategy(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, std::ostringstream& debug) {
  if(state.get_winner() != -1 || state.get_round() > 0 || state.get_players()[i].has_folded()) {
    return;
  }
  else if(state.get_active() == i) {
    auto actions = valid_actions(state, get_config().action_profile);
    int cluster = FlatClusterMap::get_instance()->cluster(state.get_round(), board, hands[i]);
    size_t regret_base_idx = _regrets.index(state, cluster);
    auto freq = calculate_strategy(_regrets, regret_base_idx, actions.size());
    int a_idx = sample_action_idx(freq);
    if(get_log_level() != BlueprintLogLevel::NONE) {
      debug << "Update strategy: " << relative_history_str(state, get_config().init_state) << "\n";
      debug << "\t" << hands[i].to_string() << ": (cluster=" << cluster << ")\n\t";
      for(int ai = 0; ai < actions.size(); ++ai) {
        debug << actions[ai].to_string() << "=" << std::setprecision(2) << std::fixed << freq[ai] << "  ";
      }
      debug << "\n";
    }

    #pragma omp critical
    _phi[_phi.index(state, cluster, a_idx)] += 1.0f;

    update_strategy(state.apply(actions[a_idx]), i, board, hands, debug);
  }
  else {
    for(Action action : valid_actions(state, get_config().action_profile)) {
      update_strategy(state.apply(action), i, board, hands, debug);
    }
  }
}

void BlueprintTrainer::on_start() {
  Logger::log(_bp_config.to_string());
}

void BlueprintTrainer::on_step(long t,int i, const std::vector<Hand>& hands, std::ostringstream& debug) {
  if(t % _bp_config.strategy_interval == 0 && t < _bp_config.preflop_threshold) {
    if(get_log_level() != BlueprintLogLevel::NONE) debug << "============== Updating strategy ==============\n";
    update_strategy(get_config().init_state, i, get_config().init_board, hands, debug);
  }
}

bool BlueprintTrainer::should_prune(long t) const {
  if(t < _bp_config.prune_thresh) return false;
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  return dist(GlobalRNG::instance()) > 0.95;
}

bool BlueprintTrainer::should_discount(long t) const {
  return _bp_config.is_discount_step(t);
}

bool BlueprintTrainer::should_snapshot(long t, long T) const {
  return _bp_config.is_snapshot_step(t, T);
}

bool BlueprintTrainer::should_log(long t) const {
  return t % _bp_config.log_interval == 0;
}

bool BlueprintTrainer::is_preflop_frozen(long t) const {
  return t > _bp_config.preflop_threshold;
}

long BlueprintTrainer::next_step(long t, long T) const {
  return std::min(std::min(_bp_config.next_discount_step(t, T), _bp_config.next_snapshot_step(t, T)), T);
}

double BlueprintTrainer::get_discount_factor(long t) const {
  long discount_interval = _bp_config.discount_interval;
  return static_cast<double>(t / discount_interval) / (t / discount_interval + 1);
}

template <class T>
void log_preflop_strategy(const StrategyStorage<T>& strat, const MCCFRConfig& config, nlohmann::json& metrics, bool phi) {
  PokerState state = config.init_state;
  Board board("2c2d2h3c3h");
  for(int p = 0; p < config.poker.n_players - 1; ++p) {
    auto actions = valid_actions(state, config.action_profile);
    PokerRange range_copy = config.init_ranges[state.get_active()];
    
    auto ranges = build_renderable_ranges(strat, config.action_profile, state, board, range_copy);
    for(Action a : actions) {
      double freq = ranges.at(a).get_range().n_combos() / MAX_COMBOS;
      std::string data_label = pos_to_str(state.get_active(), config.poker.n_players) + " " + a.to_string() + (!phi ? " (regrets)" : " (phi)");
      metrics[data_label] = freq;
    }
    state = state.apply(Action::FOLD);
  }
}

std::string BlueprintTrainer::build_wandb_metrics(long t) const {
  long avg_regret = 0;
  for(auto& r : get_strategy().data()) avg_regret += std::max(r.load(), 0); // should be sum of the maximum regret at each infoset, not sum of all regrets
  avg_regret /= t;
  std::ostringstream buf;
  buf << std::setprecision(1) << std::fixed << "t=" << t / 1'000'000.0 << "M    " << "avg_regret=" << avg_regret << "\n";
  Logger::dump(buf);

  nlohmann::json metrics = {
    {"avg_regret", static_cast<int>(avg_regret)},
    {"t (M)", static_cast<float>(t / 1'000'000.0)}
  };
  log_preflop_strategy(get_strategy(), get_config(), metrics, false);
  log_preflop_strategy(get_phi(), get_config(), metrics, true);
  return metrics.dump();
}

bool BlueprintTrainer::operator==(const BlueprintTrainer& other) const {
  return _regrets == other._regrets &&
         _phi == other._phi &&
         _bp_config == other._bp_config;
}

}

