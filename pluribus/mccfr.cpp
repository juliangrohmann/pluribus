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
#include <pluribus/debug.hpp>
#include <pluribus/mccfr.hpp>

namespace pluribus {

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
    throw std::runtime_error("Non-terminal state does not have utility.");
  }
}

BlueprintTrainerConfig::BlueprintTrainerConfig(int n_players, int n_chips, int ante) 
    : BlueprintTrainerConfig{PokerConfig{n_players, n_chips, ante}} {}

BlueprintTrainerConfig::BlueprintTrainerConfig(const PokerConfig& poker_) 
    : poker{poker_}, action_profile{BlueprintActionProfile{poker_.n_players}}, init_state{poker_} {
  for(int i = 0; i < poker_.n_players; ++i) init_ranges.push_back(PokerRange::full());
  set_iterations(BlueprintTimingConfig{}, 10'000'000);
}

std::string BlueprintTrainerConfig::to_string() const {
  std::ostringstream oss;
  oss << "================ Blueprint Trainer Config ================\n";
  oss << "Poker config: " << poker.to_string() << "\n";
  oss << "Strategy interval: " << strategy_interval << "\n";
  oss << "Preflop threshold: " << preflop_threshold << "\n";
  oss << "Snapshot interval: " << snapshot_interval << "\n";
  oss << "Prune threshold: " << prune_thresh << "\n";
  oss << "LCFR threshold: " << lcfr_thresh << "\n";
  oss << "Discount interval: " << discount_interval << "\n";
  oss << "Log interval: " << log_interval << "\n";
  oss << "Prune cutoff: " << prune_cutoff << "\n";
  oss << "Regret floor: " << regret_floor << "\n";
  oss << "Initial board: " << cards_to_str(init_board.data(), init_board.size()) << "\n";
  oss << "Initial state:\n" << init_state.to_string() << "\n";
  oss << "Initial ranges:\n";
  for(int i = 0; i < init_ranges.size(); ++ i) std::cout << "Player " << i << ": " << init_ranges[i].n_combos() << " combos\n";
  oss << "Action profile:\n" << action_profile.to_string();
  oss << "----------------------------------------------------------\n";
  return oss.str();
}

std::string action_str(const std::vector<Action>& actions) {
  std::string str;
  for(Action a : actions) {
    str += a.to_string() + ", ";
  }
  return str;
}

BlueprintTrainer::BlueprintTrainer(const BlueprintTrainerConfig& config, bool enable_wandb) 
    : _regrets{config.action_profile, 200}, _phi{config.action_profile, 169}, _config{config}, _t{1} {
  if(_config.init_state.get_players().size() != config.poker.n_players) throw std::runtime_error("Player number mismatch");
  if(enable_wandb) {
    _wb = std::unique_ptr<wandb::Session>{new wandb::Session()};
    wandb::Config wb_config = {
      {"n_players", _config.poker.n_players},
      {"n_chips", _config.poker.n_chips},
      {"ante", _config.poker.ante},
      {"strategy_interval", static_cast<int>(_config.strategy_interval)},
      {"preflop_threshold (M)", static_cast<int>(_config.preflop_threshold / 1'000'000)},
      {"snapshot_interval (M)", static_cast<int>(_config.snapshot_interval / 1'000'000)},
      {"prune_thresh (M)", static_cast<int>(_config.prune_thresh / 1'000'000)},
      {"lcfr_thresh (M)", static_cast<int>(_config.lcfr_thresh / 1'000'000)},
      {"discount_interval (M)", static_cast<int>(_config.discount_interval / 1'000'000)},
      {"log_interval (M)", static_cast<int>(_config.log_interval / 1'000'000)},
      {"prune_cutoff", _config.prune_cutoff},
      {"regret_floor", _config.regret_floor},
      {"init_board", cards_to_str(_config.init_board.data(), _config.init_board.size())},
      {"init_state_round", _config.init_state.get_round()},
      {"init_state_pot", _config.init_state.get_pot()},
      {"init_state_active", _config.init_state.get_active()}
    };
    for(int r = 0; r < 4; ++r) {
      for(int b = 0; b < _config.action_profile.n_bet_levels(r); ++b) {
        for(int p = 0; p < _config.poker.n_players; ++p) {
          std::string prof_str = action_str(_config.action_profile.get_actions(r, b, p, 150));
          wb_config["action_profile_r" + std::to_string(r) + "_b" + std::to_string(b) + "_p" + std::to_string(p)] = prof_str;
        }
      }
    }
    std::string prof_str = action_str(_config.action_profile.get_actions(0, 1, 1, 250));
    wb_config["action_profile_iso_actions"] = prof_str;
    for(int p = 0; p < _config.poker.n_players; ++p) {
      wb_config["init_range_combos_p" + std::to_string(p)] = _config.init_ranges[p].n_combos();
    }
    std::ostringstream run_name;
    run_name << _config.poker.n_players << "p_" << _config.poker.n_chips / 100 << "bb_" << _config.poker.ante << "ante_" << date_time_str();
    _wb_run = _wb->initRun({
        wandb::run::WithConfig(wb_config), wandb::run::WithProject("Pluribus"),
        wandb::run::WithRunName(run_name.str()),
        // wandb::run::WithRunID("myrunid"),
    });
  }
}

bool are_full_ranges(const std::vector<PokerRange>& ranges) {
  PokerRange full_range = PokerRange::full();
  for(const auto& r : ranges) {
    if(r != full_range) return false;
  }
  return true;
}

void BlueprintTrainer::mccfr_p(long t_plus) {
  if(!create_dir(_snapshot_dir)) throw std::runtime_error("Failed to create snapshot dir: " + _snapshot_dir.string());
  if(!create_dir(_metrics_dir)) throw std::runtime_error("Failed to create metrics dir: " + _metrics_dir.string());
  if(!create_dir(_log_dir)) throw std::runtime_error("Failed to create log dir: " + _log_dir.string());

  long T = _t + t_plus;
  std::cout << "BlueprintTrainer --- Initializing HandIndexer... " << std::flush << (HandIndexer::get_instance() ? "Success.\n" : "Failure.\n");
  std::cout << "BlueprintTrainer --- Initializing FlatClusterMap... " << std::flush << (FlatClusterMap::get_instance() ? "Success.\n" : "Failure.\n");
  std::cout << _config.to_string() << "\n";

  bool full_ranges = are_full_ranges(_config.init_ranges);
  std::cout << "Full ranges: " << (full_ranges ? "true" : "false") << "\n";
  std::cout << "Training blueprint from " << _t << " to " << std::to_string(T) << "\n";
  while(_t < T) {
    long init_t = _t;
    long next_discount = _config.next_discount_step(_t, T);
    long next_snapshot = _config.next_snapshot_step(_t, T);
    _t = std::min(std::min(next_discount, next_snapshot), T);
    auto interval_start = std::chrono::high_resolution_clock::now();
    std::cout << std::setprecision(1) << std::fixed << "Next step: " << _t / 1'000'000.0 << "M\n";
    #pragma omp parallel for schedule(dynamic, 1)
    for(long t = init_t; t < _t; ++t) {
      thread_local omp::HandEvaluator eval;
      thread_local Deck deck{_config.init_board};
      thread_local Board board;
      thread_local std::vector<Hand> hands{static_cast<size_t>(_config.poker.n_players)};
      thread_local std::ostringstream debug;
      if(_log_level != BlueprintLogLevel::NONE) debug << "============== t = " << t << " ==============\n";
      if(t % (_config.log_interval) == 0) log_metrics(t);
      for(int i = 0; i < _config.poker.n_players; ++i) {
        if(_log_level != BlueprintLogLevel::NONE) debug << "============== i = " << i << " ==============\n";
        deck.shuffle();
        board.deal(deck, _config.init_board);
        if(full_ranges) {
          for(auto& hand : hands) hand.deal(deck);
        }
        else {
          std::unordered_set<uint8_t> dead_cards;
          std::copy(board.cards().begin(), board.cards().end(), std::inserter(dead_cards, dead_cards.end()));
          for(int p_idx = 0; p_idx < _config.poker.n_players; ++p_idx) {
            throw std::runtime_error("Biased MCCFR sampling not implemented.");
            // hands[p_idx] = _config.init_ranges[p_idx].sample(dead_cards);
            dead_cards.insert(hands[p_idx].cards()[0]);
            dead_cards.insert(hands[p_idx].cards()[1]);
          }
        }

        if(t % _config.strategy_interval == 0 && t < _config.preflop_threshold) {
          if(_log_level != BlueprintLogLevel::NONE) debug << "============== Updating strategy ==============\n";
          update_strategy(_config.init_state, i, board, hands, debug);
        }
        if(t > _config.prune_thresh) {
          std::uniform_real_distribution<float> dist(0.0f, 1.0f);
          float q = dist(GlobalRNG::instance());
          if(q < 0.05f) {
            if(_log_level != BlueprintLogLevel::NONE) debug << "============== Traverse MCCFR ==============\n";
            traverse_mccfr(_config.init_state, t, i, board, hands, eval, debug);
          }
          else {
            if(_log_level != BlueprintLogLevel::NONE) debug << "============== Traverse MCCFR-P ==============\n";
            traverse_mccfr_p(_config.init_state, t, i, board, hands, eval, debug);
          }
        }
        else {
          if(_log_level != BlueprintLogLevel::NONE) debug << "============== Traverse MCCFR ==============\n";
          traverse_mccfr(_config.init_state, t, i, board, hands, eval, debug);
        }
      }
      if(_log_level == BlueprintLogLevel::DEBUG) {
        write_to_file(_log_dir / ("t" + std::to_string(t) + ".log"), debug.str());
      }
      debug.str("");
    }
    
    auto interval_end = std::chrono::high_resolution_clock::now();
    std::cout << "Step duration: " << std::chrono::duration_cast<std::chrono::seconds>(interval_end - interval_start).count() << " s.\n";
    if(_t == next_discount) {
      std::cout << "============== Discounting ==============\n";
      long discount_interval = _config.discount_interval;
      double d = static_cast<double>(_t / discount_interval) / (_t / discount_interval + 1);
      std::cout << std::setprecision(2) << std::fixed << "Discount factor: " << d << "\n";
      lcfr_discount(_regrets, d);
      lcfr_discount(_phi, d);
    }
    if(_t == next_snapshot) {
      std::ostringstream fn_stream;
      if(_t == _config.preflop_threshold) {
        std::cout << "============== Saving & freezing preflop strategy ==============\n";
        fn_stream << date_time_str() << "_preflop.bin";
      }
      else {
        std::cout << "============== Saving snapshot ==============\n";
        fn_stream << date_time_str() << "_t" << std::setprecision(1) << std::fixed << _t / 1'000'000.0 << "M.bin";
      }
      cereal_save(*this, (_snapshot_dir / fn_stream.str()).string());
    }
  }

  std::cout << "============== Blueprint training complete ==============\n";
  std::ostringstream oss;
  oss << date_time_str() << _config.poker.n_players << "p_" << _config.poker.n_chips / 100 << "bb_" << _config.poker.ante << "ante_"
      << std::setprecision(1) << std::fixed << T / 1'000'000'000.0 << "B.bin";
  cereal_save(*this, oss.str());
}

std::string info_str(const PokerState& state, int prev_r, int d_r, long t, const Board& board, const std::vector<Hand>& hands) {
  std::string str = "r=" + std::to_string(prev_r) + " + " + std::to_string(d_r) + "\nt=" + 
         std::to_string(t) + "\nBoard=" + board.to_string() + "\nHands=";
  for(const auto& hand : hands) str += hand.to_string() + "  ";
  str += "\n" + state.get_action_history().to_string() + "\n";
  return str;
}

void BlueprintTrainer::error(const std::string& msg, const std::ostringstream& debug) const {
  std::string error_msg = msg;
  if(_log_level != BlueprintLogLevel::NONE) {
    auto debug_dir = _log_dir / ("thread" + std::to_string(omp_get_thread_num()) + ".error");
    write_to_file(debug_dir, debug.str() + "\nRUNTIME ERROR: " + msg);
    error_msg += "\nDebug logs written to " + debug_dir.string();
  }
  throw std::runtime_error(error_msg);
}

int BlueprintTrainer::traverse_mccfr_p(const PokerState& state, long t, int i, const Board& board, const std::vector<Hand>& hands, 
                                       const omp::HandEvaluator& eval, std::ostringstream& debug) {
  if(state.is_terminal() || state.get_players()[i].has_folded()) {
    int u = utility(state, i, board, hands, _config.poker.n_chips, eval);
    if(_log_level != BlueprintLogLevel::NONE) log_utility(u, state, hands, debug);
    return u;
  }
  else if(state.get_active() == i) {
    auto actions = valid_actions(state, _config.action_profile);
    int cluster = FlatClusterMap::get_instance()->cluster(state.get_round(), board, hands[i]);
    if(_log_level != BlueprintLogLevel::NONE) debug << "Cluster:" << cluster << "\n";
    size_t base_idx = _regrets.index(state, cluster);
    auto freq = calculate_strategy(_regrets, base_idx, actions.size());

    std::unordered_map<Action, int> values;
    float v_exact = 0;
    for(int a_idx = 0; a_idx < actions.size(); ++a_idx) {
      Action a = actions[a_idx];
      if(_regrets[base_idx + a_idx].load() > _config.prune_cutoff) {
        int v_a = traverse_mccfr_p(state.apply(a), t, i, board, hands, eval, debug);
        values[a] = v_a;
        v_exact += freq[a_idx] * v_a;
        if(_log_level != BlueprintLogLevel::NONE) log_action_ev(a, freq[a_idx], v_a, state, debug);
      }
    }
    int v = round(v_exact);
    if(_log_level != BlueprintLogLevel::NONE) log_net_ev(v, v_exact, state, debug);
    if(abs(v) > (state.get_players().size() - 1) * _config.poker.n_chips) {
      error("EV too large!\nEV=" + std::to_string(v), debug);
    }
    if(state.get_round() == 0 && t > _config.preflop_threshold) return v;
    for(int a_idx = 0; a_idx < actions.size(); ++a_idx) {
      Action a = actions[a_idx];
      if(values.find(a) != values.end()) {
        int prev_r = _regrets[base_idx + a_idx].load();
        int d_r = values[a] - v;
        int next_r = prev_r + d_r;
        if(next_r > 2'000'000'000) error("Regret overflowing!\n" + info_str(state, prev_r, d_r, t, board, hands), debug);
        int total_r = std::max(next_r, _config.regret_floor);
        _regrets[base_idx + a_idx].store(total_r);
        if(_log_level != BlueprintLogLevel::NONE) log_regret(actions[a_idx], d_r, total_r, debug);
      }
    }
    return v;
  }
  else {
    auto actions = valid_actions(state, _config.action_profile);
    std::vector<float> freq;
    if(state.get_round() == 0 && t > _config.preflop_threshold) {
      freq = state_to_freq(state, board, hands[state.get_active()], actions.size(), _phi);
    }
    else {
      freq = state_to_freq(state, board, hands[state.get_active()], actions.size(), _regrets);
    }
    Action a = actions[sample_action_idx(freq)];
    
    return traverse_mccfr_p(state.apply(a), t, i, board, hands, eval, debug);
  }
}

std::string relative_history_str(const PokerState& state, const BlueprintTrainerConfig& config) {
  return state.get_action_history().slice(config.init_state.get_action_history().size()).to_string();
}

int BlueprintTrainer::traverse_mccfr(const PokerState& state, long t, int i, const Board& board, const std::vector<Hand>& hands, 
                                     const omp::HandEvaluator& eval, std::ostringstream& debug) {
  if(state.is_terminal() || state.get_players()[i].has_folded()) {
    int u = utility(state, i, board, hands, _config.poker.n_chips, eval);
    if(_log_level != BlueprintLogLevel::NONE) log_utility(u, state, hands, debug);
    if(abs(u) > (state.get_players().size() - 1) * _config.poker.n_chips) error("Utility too large!", debug);
    return u;
  }
  else if(state.get_active() == i) {
    auto actions = valid_actions(state, _config.action_profile);
    int cluster = FlatClusterMap::get_instance()->cluster(state.get_round(), board, hands[i]);
    if(_log_level != BlueprintLogLevel::NONE) debug << "Cluster:" << cluster << "\n";
    size_t base_idx = _regrets.index(state, cluster);
    auto freq = calculate_strategy(_regrets, base_idx, actions.size());
    std::unordered_map<Action, int> values;
    float v_exact = 0;
    for(int a_idx = 0; a_idx < actions.size(); ++a_idx) {
      Action a = actions[a_idx];
      int v_a = traverse_mccfr(state.apply(a), t, i, board, hands, eval, debug);
      values[a] = v_a;
      v_exact += freq[a_idx] * v_a;
      if(_log_level != BlueprintLogLevel::NONE) log_action_ev(a, freq[a_idx], v_a, state, debug);
    }
    int v = round(v_exact);
    if(_log_level != BlueprintLogLevel::NONE) log_net_ev(v, v_exact, state, debug);
    if(abs(v) > (state.get_players().size() - 1) * _config.poker.n_chips) {
      error("EV too large!\n" + state.get_action_history().to_string() + "\nEV=" + std::to_string(v), debug);
    }
    if(state.get_round() == 0 && t > _config.preflop_threshold) return v;

    for(int a_idx = 0; a_idx < actions.size(); ++a_idx) {
      int prev_r = _regrets[base_idx + a_idx].load();
      int d_r = values[actions[a_idx]] - v;
      int next_r = prev_r + d_r;
      if(next_r > 2'000'000'000) error("Regret overflowing!\n" + info_str(state, prev_r, d_r, t, board, hands), debug);
      int total_r = std::max(next_r, _config.regret_floor);
      _regrets[base_idx + a_idx].store(total_r);
      if(_log_level != BlueprintLogLevel::NONE) log_regret(actions[a_idx], d_r, total_r, debug);
    }
    return v;
  }
  else {
    auto actions = valid_actions(state, _config.action_profile);
    std::vector<float> freq;
    if(state.get_round() == 0 && t > _config.preflop_threshold) {
      freq = state_to_freq(state, board, hands[state.get_active()], actions.size(), _phi);
    }
    else {
      freq = state_to_freq(state, board, hands[state.get_active()], actions.size(), _regrets);
    }
    Action a = actions[sample_action_idx(freq)];
    if(_log_level != BlueprintLogLevel::NONE) log_external_sampling(a, actions, freq, state, debug);
    return traverse_mccfr(state.apply(a), t, i, board, hands, eval, debug);
  }
}

void BlueprintTrainer::update_strategy(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, std::ostringstream& debug) {
  if(state.get_winner() != -1 || state.get_round() > 0 || state.get_players()[i].has_folded()) {
    return;
  }
  else if(state.get_active() == i) {
    auto actions = valid_actions(state, _config.action_profile);
    int cluster = FlatClusterMap::get_instance()->cluster(state.get_round(), board, hands[i]);
    if(hands[i].cards()[0] % 4 == hands[i].cards()[1] % 4 && cluster < 91) {
      error("Bad cluster for suited hand: " + hands[i].to_string(), debug);
    }
    size_t regret_base_idx = _regrets.index(state, cluster);
    auto freq = calculate_strategy(_regrets, regret_base_idx, actions.size());
    int a_idx = sample_action_idx(freq);
    if(_log_level != BlueprintLogLevel::NONE) {
      debug << "Update strategy: " << relative_history_str(state, _config) << "\n";
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
    for(Action action : valid_actions(state, _config.action_profile)) {
      update_strategy(state.apply(action), i, board, hands, debug);
    }
  }
}

template <class T>
void log_preflop_strategy(const StrategyStorage<T>& strat, const BlueprintTrainerConfig& config, nlohmann::json& metrics, bool phi) {
  PokerState state = config.init_state;
  Board board("2c2d2h3c3h");
  for(int p = 0; p < config.poker.n_players - 1; ++p) {
    auto actions = valid_actions(state, config.action_profile);
    PokerRange range_copy = config.init_ranges[state.get_active()];
    
    auto ranges = build_renderable_ranges(strat, config.action_profile, state, board, range_copy);
    for(Action a : actions) {
      double freq = ranges.at(a).get_range().n_combos() / 1326.0;
      std::string data_label = pos_to_str(state.get_active(), config.poker.n_players) + " " + a.to_string() + (!phi ? " (regrets)" : " (phi)");
      metrics[data_label] = freq;
    }
    state = state.apply(Action::FOLD);
  }
}

void BlueprintTrainer::log_utility(int utility, const PokerState& state, const std::vector<Hand>& hands, std::ostringstream& debug) const {
  debug << "Terminal: " << relative_history_str(state, _config) << "\n";
  debug << "\tHands: ";
  for(int p_idx = 0; p_idx < state.get_players().size(); ++p_idx) {
    debug << hands[p_idx].to_string() << " ";
  }
  debug << "\n";
  debug << "\tu(z) = " << utility << "\n";
}

void BlueprintTrainer::log_action_ev(Action a, float freq, int ev, const PokerState& state, std::ostringstream& debug) const {
  debug << "Action EV: " << relative_history_str(state, _config) << "\n";
  debug << "\tu(" << a.to_string() << ") @ " << std::setprecision(2) << std::fixed << freq << " = " << ev << "\n";
}

void BlueprintTrainer::log_net_ev(int ev, float ev_exact, const PokerState& state, std::ostringstream& debug) const {
  debug << "Net EV: " << relative_history_str(state, _config) << "\n";
  debug << "\tu(sigma) = " << std::setprecision(2) << std::fixed << ev << " (exact=" << ev_exact << ")\n";
  if(state.get_round() == 0) debug << "Preflop frozen, skipping regret update.\n";
}

void BlueprintTrainer::log_regret(Action a, int d_r, int total_r, std::ostringstream& debug) const {
  debug << "\tR(" << a.to_string() << ") = " << d_r << "\n";
  debug << "\tcum R(" << a.to_string() << ") = " << total_r << "\n";
}

void BlueprintTrainer::log_external_sampling(Action sampled, const std::vector<Action>& actions, const std::vector<float>& freq, 
                                             const PokerState& state, std::ostringstream& debug) const {
  debug << "Sampling: " << relative_history_str(state, _config) << "\n\t";
  for(int a_idx = 0; a_idx < actions.size(); ++a_idx) {
    debug << std::setprecision(2) << std::fixed << actions[a_idx].to_string() << "=" << freq[a_idx] << " ";
  }
  debug << "\n";
  debug << "\tSampled: " << sampled.to_string() << "\n";
}

void BlueprintTrainer::log_metrics(long t) {
  long avg_regret = 0;
  for(auto& r : _regrets.data()) avg_regret += std::max(r.load(), 0); // should be sum of the maximum regret at each infoset, not sum of all regrets
  avg_regret /= t;
  std::cout << std::setprecision(1) << std::fixed << "t=" << t / 1'000'000.0 << "M    " << "avg_regret=" << avg_regret << "\n";

  nlohmann::json metrics = {
    {"avg_regret", static_cast<int>(avg_regret)},
    {"t (M)", static_cast<float>(t / 1'000'000.0)}
  };
  log_preflop_strategy(get_strategy(), get_config(), metrics, false);
  log_preflop_strategy(get_phi(), get_config(), metrics, true);
  std::ostringstream metrics_fn;
  metrics_fn << std::setprecision(1) << std::fixed << t / 1'000'000.0 << ".json";
  write_to_file(_metrics_dir / metrics_fn.str(), metrics.dump());

  if(_wb) {
    wandb::History wb_data;
    for(const auto& el : metrics.items()) {
      if(el.value().is_number_integer()) wb_data[el.key()] = el.value().get<int>();
      else if(el.value().is_number_float()) wb_data[el.key()] = el.value().get<float>();
      else if(el.value().is_string()) wb_data[el.key()] = el.value().get<std::string>();
    }
    _wb_run.log(wb_data);
  } 
}

bool BlueprintTrainer::operator==(const BlueprintTrainer& other) const {
  return _regrets == other._regrets &&
         _phi == other._phi &&
         _config == other._config &&
         _t == other._t;
}

}

