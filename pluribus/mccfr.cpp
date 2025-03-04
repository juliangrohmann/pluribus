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
#include <cereal/archives/binary.hpp>
#include <cereal/types/unordered_map.hpp>
#include <pluribus/util.hpp>
#include <pluribus/rng.hpp>
#include <pluribus/debug.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/cluster.hpp>
#include <pluribus/actions.hpp>
#include <pluribus/history_index.hpp>
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

std::string BlueprintTrainerConfig::to_string() const {
  std::ostringstream oss;
  oss << "================ Blueprint Trainer Config ================\n";
  oss << "Poker config: " << poker.to_string() << "\n";
  oss << "Strategy interval: " << strategy_interval << "\n";
  oss << "Preflop threshold: " << preflop_threshold_m << " m\n";
  oss << "Snapshot interval: " << snapshot_interval_m << " m\n";
  oss << "Prune threshold: " << prune_thresh_m << " m\n";
  oss << "LCFR threshold: " << lcfr_thresh_m << " m\n";
  oss << "Discount interval: " << discount_interval_m << " m\n";
  oss << "Log interval: " << log_interval_m << " m\n";
  oss << "Profiling threshold: " << profiling_thresh << "\n";
  oss << "Prune cutoff: " << prune_cutoff << "\n";
  oss << "Regret floor: " << regret_floor << "\n";
  oss << "----------------------------------------------------------\n";
  return oss.str();
}

BlueprintTrainer::BlueprintTrainer(const BlueprintTrainerConfig& config, const std::string& snapshot_dir) 
    : _regrets{config.poker, 200, config.action_profile.max_actions()}, _phi{}, _config{config}, _snapshot_dir{snapshot_dir}, 
      _t{1}, _it_per_min{50'000} {
  std::cout << "BlueprintTrainer --- Initializing HandIndexer... " << std::flush << (HandIndexer::get_instance() ? "Success.\n" : "Failure.\n");
  std::cout << "BlueprintTrainer --- Initializing FlatClusterMap... " << std::flush << (FlatClusterMap::get_instance() ? "Success.\n" : "Failure.\n");
  HistoryIndexer::get_instance()->initialize(_config.poker);
  std::cout << _config.to_string() << "\n";
}

void BlueprintTrainer::mccfr_p(long T) {
  if(verbose) omp_set_num_threads(1);
  long limit = std::numeric_limits<long>::max();
  long next_discount = limit;
  long next_snapshot = limit;
  std::cout << "Training blueprint from " << _t << " to " << (_t < _config.profiling_thresh ? "TBD" : std::to_string(limit)) 
            << " (" << T << " min)\n";
  while(_t < limit) {
    long init_t = _t;
    _t = _t < _config.profiling_thresh ? _config.profiling_thresh : std::min(std::min(next_discount, next_snapshot), limit);
    auto interval_start = std::chrono::high_resolution_clock::now();
    std::cout << std::setprecision(1) << std::fixed << "Next step: " << _t / 1'000'000.0 << "M\n";
    #pragma omp parallel for schedule(dynamic, 1)
    for(long t = init_t; t < _t; ++t) {
      thread_local omp::HandEvaluator eval;
      thread_local Deck deck;
      thread_local Board board;
      thread_local std::vector<Hand> hands{static_cast<size_t>(_config.poker.n_players)};
      thread_local PokerState root{_config.poker};
      if(verbose) std::cout << "============== t = " << t << " ==============\n";
      if(t % (_config.log_interval_m * _it_per_min) == 0) log_metrics(t);
      for(int i = 0; i < _config.poker.n_players; ++i) {
        if(verbose) std::cout << "============== i = " << i << " ==============\n";
        deck.shuffle();
        board.deal(deck);
        for(Hand& hand : hands) hand.deal(deck);

        if(t <= _config.preflop_threshold_m * _it_per_min && t % _config.strategy_interval == 0) {
          if(verbose) std::cout << "============== Updating strategy ==============\n";
          update_strategy(root, i, board, hands);
        }
        if(t > _config.prune_thresh_m * _it_per_min) {
          std::uniform_real_distribution<float> dist(0.0f, 1.0f);
          float q = dist(GlobalRNG::instance());
          if(q < 0.05f) {
            if(verbose) std::cout << "============== Traverse MCCFR ==============\n";
            traverse_mccfr(root, i, board, hands, eval);
          }
          else {
            if(verbose) std::cout << "============== Traverse MCCFR-P ==============\n";
            traverse_mccfr_p(root, i, board, hands, eval);
          }
        }
        else {
          if(verbose) std::cout << "============== Traverse MCCFR ==============\n";
          traverse_mccfr(root, i, board, hands, eval);
        }
      }
    }
    auto interval_end = std::chrono::high_resolution_clock::now();

    if(_t == _config.profiling_thresh) {
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(interval_end - interval_start).count();
      long it_per_sec = ((_config.profiling_thresh - init_t) / (ms / 1000.0));
      _it_per_min = 60 * it_per_sec;
      next_discount = _config.discount_interval_m * _it_per_min;
      next_snapshot = _config.preflop_threshold_m * _it_per_min;
      limit = T * _it_per_min;
      std::cout << "============== Profiling ==============\n";
      std::cout << "It/sec: " << it_per_sec << "\n";
      std::cout << std::setprecision(1) << std::fixed << "It/min: " << _it_per_min / 1'000'000.0 << "M\n"
                << "Limit: " << limit / 1'000'000'000.0 << "B\n";
    }
    else {
      std::cout << "Step duration: " << std::chrono::duration_cast<std::chrono::seconds>(interval_end - interval_start).count() << " s.\n";
      if(_t == next_discount) {
        std::cout << "============== Discounting ==============\n";
        long discount_interval = _config.discount_interval_m * _it_per_min;
        double d = static_cast<double>(_t / discount_interval) / (_t / discount_interval + 1);
        std::cout << std::setprecision(2) << std::fixed << "Discount factor: " << d << "\n";
        lcfr_discount(_regrets, d);
        lcfr_discount(_phi, d);
        next_discount = next_discount + discount_interval < _config.lcfr_thresh_m * _it_per_min ? next_discount + discount_interval : limit + 1;
      }
      if(_t == next_snapshot) {
        std::ostringstream fn_stream;
        if(_t == _config.preflop_threshold_m * _it_per_min) {
          std::cout << "============== Saving & freezing preflop strategy ==============\n";
          fn_stream << date_time_str() << "_preflop.bin";
        }
        else {
          std::cout << "============== Saving snapshot ==============\n";
          fn_stream << date_time_str() << "_t" << std::setprecision(1) << std::fixed << _t / 1'000'000.0 << "M.bin";
        }
        cereal_save(*this, (_snapshot_dir / fn_stream.str()).string());
        next_snapshot += _config.snapshot_interval_m * _it_per_min;
      }
    }
  }

  std::cout << "============== Blueprint training complete ==============\n";
  std::ostringstream oss;
  oss << date_time_str() << _config.poker.n_players << "p_" << _config.poker.n_chips / 100 << "bb_" << _config.poker.ante << "ante_"
      << std::setprecision(1) << std::fixed << limit / 1'000'000'000.0 << "B.bin";
  cereal_save(*this, oss.str());
}

int BlueprintTrainer::traverse_mccfr_p(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, 
                                       const omp::HandEvaluator& eval) {
  if(state.is_terminal()) {
    return utility(state, i, board, hands, eval);
  }
  else if(state.get_active() == i) {
    InformationSet info_set{state.get_action_history(), board, hands[i], state.get_round(), _config.poker};
    auto actions = valid_actions(state, _config.action_profile);
    auto action_regret_p = _regrets[info_set];
    auto freq = calculate_strategy(action_regret_p, actions.size());
    std::unordered_map<Action, int> values;
    int v = 0;
    for(int a_idx = 0; a_idx < actions.size(); ++a_idx) {
      Action a = actions[a_idx];
      if((action_regret_p + a_idx)->load() > _config.prune_cutoff) {
        int v_a = traverse_mccfr_p(state.apply(a), i, board, hands, eval);
        values[a] = v_a;
        v += freq[a_idx] * v_a;
      }
    }
    for(int a_idx = 0; a_idx < actions.size(); ++a_idx) {
      Action a = actions[a_idx];
      if(values.find(a) != values.end()) {
        int next_r = (action_regret_p + a_idx)->load() + values[a] - v;
        (action_regret_p + a_idx)->store(std::max(next_r, _config.regret_floor));
      }
    }
    return v;
  }
  else {
    InformationSet info_set{state.get_action_history(), board, hands[state.get_active()], state.get_round(), _config.poker};
    auto actions = valid_actions(state, _config.action_profile);
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
    InformationSet info_set{state.get_action_history(), board, hands[i], state.get_round(), _config.poker};
    auto actions = valid_actions(state, _config.action_profile);
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
      (action_regret_p + a_idx)->store(std::max(next_r, _config.regret_floor));
      if(verbose) {
        std::cout << "\t R(" << actions[a_idx].to_string() << ") = " << dR << "\n";
        std::cout << "\t cum R(" << actions[a_idx].to_string() << ") = " << (action_regret_p + a_idx)->load() << "\n";
      }
    }
    return v;
  }
  else {
    InformationSet info_set{state.get_action_history(), board, hands[state.get_active()], state.get_round(), _config.poker};
    auto actions = valid_actions(state, _config.action_profile);
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
    InformationSet info_set{state.get_action_history(), board, hands[i], state.get_round(), _config.poker};
    auto actions = valid_actions(state, _config.action_profile);
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
    for(Action action : valid_actions(state, _config.action_profile)) {
      update_strategy(state.apply(action), i, board, hands);
    }
  }
}

int BlueprintTrainer::utility(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, const omp::HandEvaluator& eval) const {
  if(state.get_winner() != -1) {
    return state.get_players()[i].get_chips() - _config.poker.n_chips + (state.get_winner() == i ? state.get_pot() : 0);
  }
  else if(state.get_round() >= 4) {
    return state.get_players()[i].get_chips() - _config.poker.n_chips + showdown_payoff(state, i, board, hands, eval);
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

bool BlueprintTrainer::operator==(const BlueprintTrainer& other) const {
  return _regrets == other._regrets &&
         _phi == other._phi &&
         _config == other._config &&
         _t == other._t &&
         _it_per_min == other._it_per_min;
}

}

