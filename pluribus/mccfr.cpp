#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <tqdm/tqdm.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/unordered_map.hpp>
#include <pluribus/rng.hpp>
#include <pluribus/debug.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/mccfr.hpp>

namespace pluribus {

bool StrategyState::operator==(const StrategyState& other) const {
  return regret == other.regret && frequency == other.frequency && phi == other.phi;
}

std::string minutes_str(int val) {
  return " (" + std::to_string(val / it_per_min) + " m)";
}

BlueprintTrainer::BlueprintTrainer(int n_players, int n_chips, int ante, int strategy_interval, int preflop_threshold, int snapshot_interval, 
                                   int prune_thresh, int prune_cutoff, int regret_floor, int lcfr_thresh, int discount_interval, int log_interval) : 
    _strategy{}, _eval{}, _n_players{n_players}, _n_chips{n_chips}, _ante{ante}, _strategy_interval{strategy_interval}, 
    _preflop_threshold{preflop_threshold}, _snapshot_interval{snapshot_interval}, _prune_thresh{prune_thresh}, _prune_cutoff{prune_cutoff},
    _regret_floor{regret_floor}, _lcfr_thresh{lcfr_thresh}, _discount_interval{discount_interval}, _log_interval{log_interval} {
  std::cout << "N players: " << _n_players << "\n";
  std::cout << "N chips: " << _n_chips << "\n";
  std::cout << "Ante: " << _ante << "\n";
  std::cout << "Iterations/s: " << it_per_sec << "\n";
  std::cout << "Iterations/min: " << it_per_min << "\n";
  std::cout << "Strategy interval: " << _strategy_interval << "\n";
  std::cout << "Preflop threshold: " << _preflop_threshold << minutes_str(_preflop_threshold) << "\n";
  std::cout << "Snapshot interval: " << _snapshot_interval << minutes_str(_snapshot_interval) << "\n";
  std::cout << "Prune threshold: " << _prune_thresh << minutes_str(_prune_thresh) << "\n";
  std::cout << "Prune cutoff: " << _prune_cutoff << "\n";
  std::cout << "Regret floor: " << _regret_floor << "\n";
  std::cout << "LCFR threshold: " << _lcfr_thresh << minutes_str(_lcfr_thresh) << "\n";
  std::cout << "Discount interval: " << _discount_interval << minutes_str(_discount_interval) << "\n";
  std::cout << "Log interval: " << _log_interval << minutes_str(_log_interval) << "\n";
}

void BlueprintTrainer::mccfr_p(long T) {
  Deck deck;
  Board board;
  std::vector<Hand> hands{static_cast<size_t>(_n_players)};
  for(long t = 1; t < T + 1; ++t) {
    if(verbose) std::cout << "============== t = " << t << " ==============\n";
    if(t % _log_interval == 0) log_metrics(t);
    for(int i = 0; i < _n_players; ++i) {
      if(verbose) std::cout << "============== i = " << i << " ==============\n";
      deck.shuffle();
      board.deal(deck);
      for(Hand& hand : hands) hand.deal(deck);

      if(t <= _preflop_threshold && t % _strategy_interval == 0) {
        if(verbose) std::cout << "============== Updating strategy ==============\n";
        update_strategy(PokerState{_n_players, _n_chips, _ante}, i, board, hands);
      }
      if(t > _prune_thresh) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float q = dist(GlobalRNG::instance());
        if(q < 0.05f) {
          if(verbose) std::cout << "============== Traverse MCCFR ==============\n";
          traverse_mccfr(PokerState{_n_players, _n_chips, _ante}, i, board, hands);
        }
        else {
          if(verbose) std::cout << "============== Traverse MCCFR-P ==============\n";
          traverse_mccfr_p(PokerState{_n_players, _n_chips, _ante}, i, board, hands);
        }
      }
      else {
        if(verbose) std::cout << "============== Traverse MCCFR ==============\n";
        traverse_mccfr(PokerState{_n_players, _n_chips, _ante}, i, board, hands);
      }
    }
    if(t < _lcfr_thresh && t % _discount_interval == 0) {
      if(verbose) std::cout << "============== Discounting ==============\n";
      double d = (t / _discount_interval) / (t / _discount_interval + 1);
      if(verbose) std::cout << "Discount factor: " << d << "\n";
      for(auto& info_entry : _strategy) {
        for(auto& action_entry : info_entry.second) {
          action_entry.second.regret *= d;
          action_entry.second.phi *= d;
        }
      }
    }
    if(t == _preflop_threshold) {
      if(verbose) std::cout << "============== Saving & freezing preflop strategy ==============\n";
      std::ostringstream oss;
      oss << date_time_str() << "_preflop.bin";
      save_strategy(oss.str());
    }
    else if(t > _preflop_threshold && t % _snapshot_interval == 0) {
      if(verbose) std::cout << "============== Saving snapshot ==============\n";
      std::ostringstream oss;
      oss << date_time_str() << "_t" << std::fixed << std::setprecision(1) << static_cast<double>(t) / 1'000'000 << "M.bin";
      save_strategy(oss.str());
    }
  }
}

int BlueprintTrainer::traverse_mccfr_p(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands) {
  if(state.is_terminal()) {
    return utility(state, i, board, hands);
  }
  else if(state.get_active() == i) {
    InformationSet info_set{state.get_action_history(), board, hands[i], state.get_round()};
    calculate_strategy(state, info_set);
    std::unordered_map<Action, bool> explored;
    std::unordered_map<Action, int> values;
    int v = 0;
    for(const auto& entry : _strategy.at(info_set)) {
      if(entry.second.regret > _prune_cutoff) {
        int v_a = traverse_mccfr_p(state.apply(entry.first), i, board, hands);
        values[entry.first] = v_a;
        explored[entry.first] = true;
        v += entry.second.frequency * v_a;
      }
      else {
        explored[entry.first] = false;
      }
    }
    for(auto& entry : _strategy.at(info_set)) {
      if(explored[entry.first]) {
        entry.second.regret = std::max(entry.second.regret + values[entry.first] - v, _regret_floor);
      }
    }
    return v;
  }
  else {
    InformationSet info_set{state.get_action_history(), board, hands[state.get_active()], state.get_round()};
    calculate_strategy(state, info_set);
    Action action = sample(info_set);
    return traverse_mccfr_p(state.apply(action), i, board, hands);
  }
}

int BlueprintTrainer::traverse_mccfr(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands) {
  if(state.is_terminal()) {
    int u = utility(state, i, board, hands);
    if(verbose) {
      std::cout << state.get_action_history().to_string() << "\n";
      std::cout << "\tu(z) = " << u << "\n";
    }
    return u;
  }
  else if(state.get_active() == i) {
    InformationSet info_set{state.get_action_history(), board, hands[i], state.get_round()};
    calculate_strategy(state, info_set);
    std::unordered_map<Action, int> values;
    int v = 0;
    for(const auto& entry : _strategy.at(info_set)) {
      int v_a = traverse_mccfr(state.apply(entry.first), i, board, hands);
      values[entry.first] = v_a;
      v += entry.second.frequency * v_a;
      if(verbose) {
        std::cout << state.get_action_history().to_string() << "\n";
        std::cout << "\t u(" << action_to_str.at(entry.first) << ") @ " << entry.second.frequency << " = " << v_a << "\n";
      }
    }
    if(verbose) {
      std::cout << state.get_action_history().to_string() << "\n";
      std::cout << "\t u(sigma) = " << v << "\n";
    }
    for(auto& entry : _strategy.at(info_set)) {
      int dR = values[entry.first] - v;
      entry.second.regret = std::max(entry.second.regret + dR, _regret_floor);
      if(verbose) {
        std::cout << "\t R(" << action_to_str.at(entry.first) << ") = " << dR << "\n";
        std::cout << "\t cum R(" << action_to_str.at(entry.first) << ") = " << entry.second.regret << "\n";
      }
    }
    return v;
  }
  else {
    InformationSet info_set{state.get_action_history(), board, hands[state.get_active()], state.get_round()};
    calculate_strategy(state, info_set);
    Action action = sample(info_set);
    return traverse_mccfr(state.apply(action), i, board, hands);
  }
}

void BlueprintTrainer::update_strategy(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands) {
  if(state.get_winner() != -1 || state.get_round() > 0 || state.get_players()[i].has_folded()) {
    return;
  }
  else if(state.get_active() == i) {
    InformationSet info_set{state.get_action_history(), board, hands[i], state.get_round()};
    calculate_strategy(state, info_set);
    Action action = sample(info_set);
    _strategy.at(info_set).at(action).phi += 1.0f;
    update_strategy(state.apply(action), i, board, hands);
  }
  else {
    for(Action action : valid_actions(state)) {
      update_strategy(state.apply(action), i, board, hands);
    }
  }
}

void BlueprintTrainer::calculate_strategy(const PokerState& state, const InformationSet& info_set) {
  int sum = 0;
  std::vector<Action> actions = valid_actions(state);
  auto& action_map = _strategy[info_set];
  for(Action action : actions) {
    sum += std::max(action_map[action].regret, 0);
  }
  if(sum > 0) {
    for(Action a : actions) {
      StrategyState& strat = action_map[a];
      strat.frequency = std::max(strat.regret, 0) / static_cast<double>(sum);
    }
  }
  else {
    for(Action a : actions) {
      action_map[a].frequency = 1.0 / actions.size();
    }
  }
}

int BlueprintTrainer::utility(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands) const {
  if(state.get_winner() != -1) {
    return state.get_players()[i].get_chips() - _n_chips + (state.get_winner() == i ? state.get_pot() : 0);
  }
  else if(state.get_round() >= 4) {
    return state.get_players()[i].get_chips() - _n_chips + showdown_payoff(state, i, board, hands);
  }
  else {
    throw std::runtime_error("Non-terminal state does not have utility.");
  }
}

int BlueprintTrainer::showdown_payoff(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands) const {
  if(state.get_players()[i].has_folded()) return 0;
  std::vector<uint8_t> win_idxs = winners(state, hands, board, _eval);
  return std::find(win_idxs.begin(), win_idxs.end(), i) != win_idxs.end() ? state.get_pot() / win_idxs.size() : 0;
}

Action BlueprintTrainer::sample(const InformationSet& info_set) const {
  std::vector<Action> actions;
  std::vector<float> weights;
  for(const auto& entry : _strategy.at(info_set)) {
    actions.push_back(entry.first);
    weights.push_back(entry.second.frequency);
  }
  if(verbose) {
    std::cout << "Sampling: ";
    for(int idx = 0; idx < actions.size(); ++idx) {
      std::cout << action_to_str.at(actions[idx]) << " = " << weights[idx] << (idx == actions.size() - 1 ? "\n" : ", ");
    }
  }
  std::discrete_distribution<> dist(weights.begin(), weights.end());
  return actions[dist(GlobalRNG::instance())];
}

void BlueprintTrainer::log_metrics(int t) {
  std::cout << std::setprecision(2) << "t=" << t << "\n";
}

void BlueprintTrainer::save_strategy(std::string fn) const {
  std::cout << "Saving strategy to " << fn << '\n';
  {
    std::ofstream os(fn, std::ios::binary);
    cereal::BinaryOutputArchive oarchive(os);
    oarchive(_strategy);
  }
}

void BlueprintTrainer::load_strategy(std::string fn) {
  std::cout << "Loading strategy from " << fn << '\n';
  {
    std::ifstream is(fn, std::ios::binary);
    cereal::BinaryInputArchive iarchive(is);
    iarchive(_strategy);
  }
}

long count(const PokerState& state) {
  if(state.is_terminal()) return 0;
  long c = state.get_round() == 0 ? 169 : 200;
  for(Action a : valid_actions(state)) {
    c += count(state.apply(a));
  }
  return c;
}

long BlueprintTrainer::count_infosets() {
  PokerState state{_n_players, _n_chips, _ante};
  return count(state);
}

}

