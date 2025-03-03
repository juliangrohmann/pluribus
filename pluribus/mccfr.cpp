#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <atomic>
#include <limits>
#include <omp.h>
#include <tqdm/tqdm.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/unordered_map.hpp>
#include <pluribus/util.hpp>
#include <pluribus/rng.hpp>
#include <pluribus/debug.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/cluster.hpp>
#include <pluribus/actions.hpp>
#include <pluribus/mccfr.hpp>

namespace pluribus {

std::vector<float> calculate_strategy(const std::atomic<int>* regret_p, int n_actions) {
  int sum = 0;
  for(int a_idx = 0; a_idx < n_actions; ++a_idx) {
    sum += std::max((regret_p + a_idx)->load(), 0);
  }

  std::vector<float> freq;
  freq.reserve(n_actions);
  if(sum > 0) {
    for(int a_idx = 0; a_idx < n_actions; ++a_idx) {
      freq.push_back(std::max((regret_p + a_idx)->load(), 0) / static_cast<double>(sum));
    }
  }
  else {
    for(int i = 0; i < n_actions; ++i) {
      freq.push_back(1.0 / n_actions);
    }
  }
  return freq;
}

int sample_action_idx(const std::vector<float>& freq) {
  std::discrete_distribution<> dist(freq.begin(), freq.end());
  return dist(GlobalRNG::instance());
}

void lcfr_discount(RegretStorage& data, double d) {
  for(auto it = data.data(); it != data.data() + data.size(); ++it) {
    it->store(it->load() * d);
  }
}

void lcfr_discount(PreflopMap& data, double d) {
  for(auto& entry : data) {
    for(int i = 0; i < entry.second.size(); ++i) {
      entry.second[i] *= d;
    }
  }
}

BlueprintTrainer::BlueprintTrainer(int n_players, int n_chips, int ante, long strategy_interval, long preflop_threshold_m, long snapshot_interval_m,
                                   long prune_thresh_m, int prune_cutoff, int regret_floor, long lcfr_thresh_m, long discount_interval_m, 
                                   long log_interval_m, long profiling_thresh, std::string snapshot_dir) : 
    _t{1}, _regrets{n_players, n_chips, ante, 200, 5}, _phi{}, _action_profile{BlueprintActionProfile{}}, _n_players{n_players}, _n_chips{n_chips}, 
    _ante{ante}, _strategy_interval{strategy_interval}, _preflop_threshold_m{preflop_threshold_m}, _snapshot_interval_m{snapshot_interval_m},
    _prune_thresh_m{prune_thresh_m}, _prune_cutoff{prune_cutoff}, _regret_floor{regret_floor}, _lcfr_thresh_m{lcfr_thresh_m},
    _discount_interval_m{discount_interval_m}, _log_interval_m{log_interval_m}, _profiling_thresh{profiling_thresh}, _it_per_min{50'000},
    _snapshot_dir{snapshot_dir} {
  std::cout << "BlueprintTrainer --- Initializing HandIndexer... " << std::flush << (HandIndexer::get_instance() ? "Success.\n" : "Failure.\n");
  std::cout << "BlueprintTrainer --- Initializing FlatClusterMap... " << std::flush << (FlatClusterMap::get_instance() ? "Success.\n" : "Failure.\n");
  HistoryIndexer::get_instance()->initialize(n_players, n_chips, ante);
  log_state();
}

void BlueprintTrainer::mccfr_p(long T) {
  if(verbose) omp_set_num_threads(1);
  long limit = std::numeric_limits<long>::max();
  long next_discount = limit;
  long next_snapshot = limit;
  std::cout << "Training blueprint from " << _t << " to " << (_t < _profiling_thresh ? "TBD" : std::to_string(limit)) << " (" << T << " min)\n";
  while(_t < limit) {
    long init_t = _t;
    _t = _t < _profiling_thresh ? _profiling_thresh : std::min(std::min(next_discount, next_snapshot), limit);
    auto interval_start = std::chrono::high_resolution_clock::now();
    std::cout << std::setprecision(1) << std::fixed << "Next step: " << _t / 1'000'000.0 << "M\n";
    #pragma omp parallel for schedule(dynamic, 1)
    for(long t = init_t; t < _t; ++t) {
      thread_local omp::HandEvaluator eval;
      thread_local Deck deck;
      thread_local Board board;
      thread_local std::vector<Hand> hands{static_cast<size_t>(_n_players)};
      if(verbose) std::cout << "============== t = " << t << " ==============\n";
      if(t % (_log_interval_m * _it_per_min) == 0) log_metrics(t);
      for(int i = 0; i < _n_players; ++i) {
        if(verbose) std::cout << "============== i = " << i << " ==============\n";
        deck.shuffle();
        board.deal(deck);
        for(Hand& hand : hands) hand.deal(deck);

        if(t <= _preflop_threshold_m * _it_per_min && t % _strategy_interval == 0) {
          if(verbose) std::cout << "============== Updating strategy ==============\n";
          update_strategy(PokerState{_n_players, _n_chips, _ante}, i, board, hands);
        }
        if(t > _prune_thresh_m * _it_per_min) {
          std::uniform_real_distribution<float> dist(0.0f, 1.0f);
          float q = dist(GlobalRNG::instance());
          if(q < 0.05f) {
            if(verbose) std::cout << "============== Traverse MCCFR ==============\n";
            traverse_mccfr(PokerState{_n_players, _n_chips, _ante}, i, board, hands, eval);
          }
          else {
            if(verbose) std::cout << "============== Traverse MCCFR-P ==============\n";
            traverse_mccfr_p(PokerState{_n_players, _n_chips, _ante}, i, board, hands, eval);
          }
        }
        else {
          if(verbose) std::cout << "============== Traverse MCCFR ==============\n";
          traverse_mccfr(PokerState{_n_players, _n_chips, _ante}, i, board, hands, eval);
        }
      }
    }
    auto interval_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(interval_end - interval_start);

    if(_t == _profiling_thresh) {
      long it_per_sec = ((_profiling_thresh - init_t) / duration.count());
      _it_per_min = 60 * it_per_sec;
      next_discount = _discount_interval_m * _it_per_min;
      next_snapshot = _preflop_threshold_m * _it_per_min;
      limit = T * _it_per_min;
      std::cout << "============== Profiling ==============\n";
      std::cout << "It/sec: " << it_per_sec << "\n";
      std::cout << std::setprecision(1) << std::fixed << "It/min: " << _it_per_min / 1'000'000.0 << "M\n"
                << "Limit: " << limit / 1'000'000'000.0 << "B\n";
    }
    else {
      std::cout << "Step duration: " << duration.count() << " s.\n";
      if(_t == next_discount) {
        std::cout << "============== Discounting ==============\n";
        long discount_interval = _discount_interval_m * _it_per_min;
        double d = static_cast<double>(_t / discount_interval) / (_t / discount_interval + 1);
        std::cout << std::setprecision(2) << std::fixed << "Discount factor: " << d << "\n";
        lcfr_discount(_regrets, d);
        lcfr_discount(_phi, d);
        next_discount = next_discount + discount_interval < _lcfr_thresh_m * _it_per_min ? next_discount + discount_interval : limit + 1;
      }
      if(_t == next_snapshot) {
        std::ostringstream fn_stream;
        if(_t == _preflop_threshold_m * _it_per_min) {
          std::cout << "============== Saving & freezing preflop strategy ==============\n";
          fn_stream << date_time_str() << "_preflop.bin";
        }
        else {
          std::cout << "============== Saving snapshot ==============\n";
          fn_stream << date_time_str() << "_t" << std::setprecision(1) << std::fixed << _t / 1'000'000.0 << "M.bin";
        }
        cereal_save(*this, (_snapshot_dir / fn_stream.str()).string());
        next_snapshot += _snapshot_interval_m * _it_per_min;
      }
    }
  }

  std::cout << "============== Blueprint training complete ==============\n";
  std::ostringstream oss;
  oss << date_time_str() << _n_players << "p_" << _n_chips / 100 << "bb_" << _ante << "ante_"
      << std::setprecision(1) << std::fixed << limit / 1'000'000'000.0 << "B.bin";
  cereal_save(*this, oss.str());
}

int BlueprintTrainer::traverse_mccfr_p(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, const omp::HandEvaluator& eval) {
  if(state.is_terminal()) {
    return utility(state, i, board, hands, eval);
  }
  else if(state.get_active() == i) {
    InformationSet info_set{state.get_action_history(), board, hands[i], state.get_round(), _n_players, _n_chips, _ante};
    auto actions = valid_actions(state, _action_profile);
    auto action_regret_p = _regrets[info_set];
    auto freq = calculate_strategy(action_regret_p, actions.size());
    std::unordered_map<Action, int> values;
    int v = 0;
    for(int a_idx = 0; a_idx < actions.size(); ++a_idx) {
      Action a = actions[a_idx];
      if((action_regret_p + a_idx)->load() > _prune_cutoff) {
        int v_a = traverse_mccfr_p(state.apply(a), i, board, hands, eval);
        values[a] = v_a;
        v += freq[a_idx] * v_a;
      }
    }
    for(int a_idx = 0; a_idx < actions.size(); ++a_idx) {
      Action a = actions[a_idx];
      if(values.find(a) != values.end()) {
        int next_r = (action_regret_p + a_idx)->load() + values[a] - v;
        (action_regret_p + a_idx)->store(std::max(next_r, _regret_floor));
      }
    }
    return v;
  }
  else {
    InformationSet info_set{state.get_action_history(), board, hands[state.get_active()], state.get_round(), _n_players, _n_chips, _ante};
    auto actions = valid_actions(state, _action_profile);
    auto freq = calculate_strategy(_regrets[info_set], actions.size());
    Action a = actions[sample_action_idx(freq)];
    return traverse_mccfr_p(state.apply(a), i, board, hands, eval);
  }
}

int BlueprintTrainer::traverse_mccfr(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, const omp::HandEvaluator& eval) {
  if(state.is_terminal()) {
    int u = utility(state, i, board, hands, eval);
    if(verbose) {
      std::cout << state.get_action_history().to_string() << "\n";
      std::cout << "\tu(z) = " << u << "\n";
    }
    return u;
  }
  else if(state.get_active() == i) {
    InformationSet info_set{state.get_action_history(), board, hands[i], state.get_round(), _n_players, _n_chips, _ante};
    auto actions = valid_actions(state, _action_profile);
    auto freq = calculate_strategy(_regrets[info_set], actions.size());
    std::unordered_map<Action, int> values;
    int v = 0;
    for(int a_idx = 0; a_idx < actions.size(); ++a_idx) {
      Action a = actions[a_idx];
      int v_a = traverse_mccfr(state.apply(a), i, board, hands, eval);
      values[a] = v_a;
      v += freq[a_idx] * v_a;
      if(verbose) {
        std::cout << state.get_action_history().to_string() << "\n";
        std::cout << "\t u(" << a.to_string() << ") @ " << freq[a_idx] << " = " << v_a << "\n";
      }
    }
    if(verbose) {
      std::cout << state.get_action_history().to_string() << "\n";
      std::cout << "\t u(sigma) = " << v << "\n";
    }
    auto action_regret_p = _regrets[info_set];
    for(int a_idx = 0; a_idx < actions.size(); ++a_idx) {
      int dR = values[actions[a_idx]] - v;
      int next_r = (action_regret_p + a_idx)->load() + dR;
      (action_regret_p + a_idx)->store(std::max(next_r, _regret_floor));
      if(verbose) {
        std::cout << "\t R(" << actions[a_idx].to_string() << ") = " << dR << "\n";
        std::cout << "\t cum R(" << actions[a_idx].to_string() << ") = " << (action_regret_p + a_idx)->load() << "\n";
      }
    }
    return v;
  }
  else {
    InformationSet info_set{state.get_action_history(), board, hands[state.get_active()], state.get_round(), _n_players, _n_chips, _ante};
    auto actions = valid_actions(state, _action_profile);
    auto freq = calculate_strategy(_regrets[info_set], actions.size());
    Action a = actions[sample_action_idx(freq)];
    return traverse_mccfr(state.apply(a), i, board, hands, eval);
  }
}

void BlueprintTrainer::update_strategy(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands) {
  if(state.get_winner() != -1 || state.get_round() > 0 || state.get_players()[i].has_folded()) {
    return;
  }
  else if(state.get_active() == i) {
    InformationSet info_set{state.get_action_history(), board, hands[i], state.get_round(), _n_players, _n_chips, _ante};
    auto actions = valid_actions(state, _action_profile);
    auto freq = calculate_strategy(_regrets[info_set], actions.size());
    int a_idx = sample_action_idx(freq);
    auto& phi_regrets = _phi[info_set];
    if(phi_regrets.size() == 0) {
      phi_regrets.grow_by(actions.size());
    }
    #pragma omp critical
    phi_regrets[a_idx] += 1.0f;
    update_strategy(state.apply(actions[a_idx]), i, board, hands);
  }
  else {
    for(Action action : valid_actions(state, _action_profile)) {
      update_strategy(state.apply(action), i, board, hands);
    }
  }
}

int BlueprintTrainer::utility(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, const omp::HandEvaluator& eval) const {
  if(state.get_winner() != -1) {
    return state.get_players()[i].get_chips() - _n_chips + (state.get_winner() == i ? state.get_pot() : 0);
  }
  else if(state.get_round() >= 4) {
    return state.get_players()[i].get_chips() - _n_chips + showdown_payoff(state, i, board, hands, eval);
  }
  else {
    throw std::runtime_error("Non-terminal state does not have utility.");
  }
}

int BlueprintTrainer::showdown_payoff(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, const omp::HandEvaluator& eval) const {
  if(state.get_players()[i].has_folded()) return 0;
  std::vector<uint8_t> win_idxs = winners(state, hands, board, eval);
  return std::find(win_idxs.begin(), win_idxs.end(), i) != win_idxs.end() ? state.get_pot() / win_idxs.size() : 0;
}

void BlueprintTrainer::log_metrics(long t) const {
  std::cout << std::setprecision(1) << std::fixed << "t=" << t / 1'000'000.0 << "M\n";
}

void BlueprintTrainer::log_state() const {
  std::cout << "N players: " << _n_players << "\n";
  std::cout << "N chips: " << _n_chips << "\n";
  std::cout << "Ante: " << _ante << "\n";
  std::cout << "Iterations/min: " << _it_per_min << "\n";
  std::cout << "Strategy interval: " << _strategy_interval << "\n";
  std::cout << "Preflop threshold: " << _preflop_threshold_m << " m\n";
  std::cout << "Snapshot interval: " << _snapshot_interval_m << " m\n";
  std::cout << "Prune threshold: " << _prune_thresh_m << " m\n";
  std::cout << "Prune cutoff: " << _prune_cutoff << "\n";
  std::cout << "Regret floor: " << _regret_floor << "\n";
  std::cout << "LCFR threshold: " << _lcfr_thresh_m << " m\n";
  std::cout << "Discount interval: " << _discount_interval_m << " m\n";
  std::cout << "Log interval: " << _log_interval_m << " m\n";
}

bool BlueprintTrainer::operator==(const BlueprintTrainer& other) {
  return _regrets == other._regrets &&
         _phi == other._phi;
         _t == other._t &&
         _strategy_interval == other._strategy_interval &&
         _preflop_threshold_m == other._preflop_threshold_m &&
         _snapshot_interval_m == other._snapshot_interval_m &&
         _prune_thresh_m == other._prune_thresh_m &&
         _lcfr_thresh_m == other._lcfr_thresh_m &&
         _discount_interval_m == other._discount_interval_m &&
         _log_interval_m == other._log_interval_m &&
         _prune_cutoff == other._prune_cutoff &&
         _regret_floor == other._regret_floor &&
         _n_players == other._n_players &&
         _n_chips == other._n_chips &&
         _ante == other._ante;
}

}

