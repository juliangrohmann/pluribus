#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <cereal/cereal.hpp>
#include <pluribus/util.hpp>
#include <pluribus/debug.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/actions.hpp>

namespace pluribus {

const Action Action::UNDEFINED{-8.0f};
const Action Action::BIAS_DUMMY{-7.0f};
const Action Action::BIAS_FOLD{-6.0f};
const Action Action::BIAS_CALL{-5.0f};
const Action Action::BIAS_RAISE{-4.0f};
const Action Action::BIAS_NONE{-3.0f};
const Action Action::ALL_IN{-2.0f};
const Action Action::FOLD{-1.0f};
const Action Action::CHECK_CALL{0.0f};

std::string Action::to_string() const {
  if(_bet_type == UNDEFINED._bet_type) return "Undefined";
  if(_bet_type == BIAS_DUMMY._bet_type) return "Bias dummy";
  if(_bet_type == BIAS_FOLD._bet_type) return "Bias: Fold";
  if(_bet_type == BIAS_CALL._bet_type) return "Bias: Call";
  if(_bet_type == BIAS_RAISE._bet_type) return "Bias: Raise";
  if(_bet_type == BIAS_NONE._bet_type) return "Bias: None";
  if(_bet_type == ALL_IN._bet_type) return "All-in";
  if(_bet_type == FOLD._bet_type) return "Fold";
  if(_bet_type == CHECK_CALL._bet_type) return "Check/Call";
  std::ostringstream oss; 
  oss << "Bet " << std::setprecision(0) << std::fixed << _bet_type * 100.0f << "%";
  return oss.str();
}

bool is_bias(const Action a) {
  return a == Action::BIAS_FOLD || a == Action::BIAS_CALL ||
      a == Action::BIAS_RAISE || a == Action::BIAS_NONE;
}

ActionHistory ActionHistory::slice(const int start, const int end) const {
  return ActionHistory{std::vector<Action>{_history.begin() + start, end != -1 ? _history.begin() + end : _history.end()}}; 
}

bool ActionHistory::is_consistent(const ActionHistory& other) const {
  for(int i = 0; i < std::min(size(), other.size()); ++i) {
    if(get(i) != other.get(i)) return false;
  }
  return true;
}

std::string ActionHistory::to_string() const {
  std::string str;
  for(int i = 0; i < _history.size(); ++i) {
    str += _history[i].to_string() + (i == _history.size() - 1 ? "" : ", ");
  }
  return str;
}

void ActionProfile::set_actions(const std::vector<Action>& actions, const int round, const int bet_level, const int pos) {
  if(bet_level >= _profile[round].size()) _profile[round].resize(bet_level + 1);
  if(pos >= _profile[round][bet_level].size()) _profile[round][bet_level].resize(pos + 1);
  _profile[round][bet_level][pos] = actions;
}

const std::vector<Action>& ActionProfile::get_actions(const int round, const int bet_level, const int pos, const int pot) const {
  if(round == 0 && bet_level == 1 && pot > 150) return _iso_actions;
  const int level_idx = std::min(bet_level, static_cast<int>(_profile[round].size()) - 1);
  const int pos_idx = std::min(pos, static_cast<int>(_profile[round][level_idx].size()) - 1);
  return _profile[round][level_idx][pos_idx]; 
}

void ActionProfile::add_action(const Action& action, const int round, const int bet_level, const int pos) {
  if(bet_level >= _profile[round].size()) _profile[round].resize(bet_level + 1);
  if(pos >= _profile[round][bet_level].size()) _profile[round][bet_level].resize(pos + 1);
  _profile[round][bet_level][pos].push_back(action);
}

std::unordered_set<Action> ActionProfile::all_actions() const {
  std::unordered_set<Action> actions;
  for(auto& round : _profile) {
    for(auto& level : round) {
      for(auto& pos : level) {
        for(Action a : pos) {
          actions.insert(a);
        }
      }
    }
  }
  return actions;
}

int ActionProfile::max_actions() const {
  int ret = 0;
  for(auto& round : _profile) {
    for(auto& level : round) {
      for(auto& pos : level) {
        ret = std::max(static_cast<int>(pos.size()), ret);
      }
    }
  }
  return std::max(ret, static_cast<int>(_iso_actions.size()));
}

std::string ActionProfile::to_string() const {
  std::ostringstream oss;
  for(int round = 0; round < 4; ++round) {
    oss << round_to_str(round) << " action profile:\n";
    for(int bet_level = 0; bet_level < _profile[round].size(); ++bet_level) {
      oss << "\tBet level " << bet_level << ":\n";
      for(int pos = 0; pos < _profile[round][bet_level].size(); ++pos) {
        oss << "\t\t" << "Position " << pos << ":  ";
        for(Action a : _profile[round][bet_level][pos]) {
          oss << a.to_string() << "  ";
        }
        oss << "\n";
      }
    }
  }
  return oss.str();
}

BlueprintActionProfile::BlueprintActionProfile(const int n_players, const int stack_size) {
  if(n_players > 2) { // preflop RFI & isos
    for(int pos = 2; pos < n_players - 2; ++pos) {
      set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.40f}, Action::ALL_IN}, 0, 1, pos);
    }
    if(n_players > 3) set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.52f}, Action::ALL_IN}, 0, 1, n_players - 2);
    set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.60f}, Action::ALL_IN}, 0, 1, n_players - 1);
    set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.80f}, Action::ALL_IN}, 0, 1, 0);
    set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.80f}, Action::ALL_IN}, 0, 1, 1);
    set_iso_actions({Action::FOLD, Action::CHECK_CALL, Action{0.80f}, Action{1.00f}, Action{2.00f}, Action::ALL_IN});
  }
  else {
    set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.75f}, Action::ALL_IN}, 0, 1, 0);
    set_iso_actions({Action::FOLD, Action::CHECK_CALL, Action{1.00f}, Action{2.00f}, Action::ALL_IN});
  }
  
  if(n_players > 2) { // preflop 3-bet
    for(int pos = 2; pos < n_players; ++pos) {
      set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.80f}, Action{1.00f}, Action{1.20f}}, 0, 2, pos);
    }
    set_actions({Action::FOLD, Action::CHECK_CALL, Action{1.00f}, Action{1.20f}, Action{1.40f}}, 0, 2, 0);
    set_actions({Action::FOLD, Action::CHECK_CALL, Action{1.00f}, Action{1.20f}, Action{1.40f}}, 0, 2, 1);
  }
  else {
    set_actions({Action::FOLD, Action::CHECK_CALL, Action{1.00f}, Action{1.333f}, Action{1.667f}, Action{2.00f}}, 0, 2, 0);
    if(stack_size < 10'000) add_action(Action{0.75f}, 0, 2, 0);
  }

  // preflop 4-bet+
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.60f}, Action{0.70f}, Action{0.80f}, Action{0.90f}, Action{1.00f}, Action::ALL_IN}, 0, 3, 0);
  if(stack_size < 10'000) add_action(Action{0.50f}, 0, 3, 0);


  // flop
  if(n_players > 2) {
    set_actions({Action::CHECK_CALL, Action{0.33f}, Action{0.50f}, Action{0.75f}, Action{1.00f}, Action::ALL_IN}, 1, 0, 0);
    set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.50f}, Action{0.75f}, Action{1.00f}, Action::ALL_IN}, 1, 1, 0);
  }
  else {
    set_actions({Action::CHECK_CALL, Action{0.16f}, Action{0.33f}, Action{0.50f}, Action{0.75f}, Action{1.00f}, Action::ALL_IN}, 1, 0, 0); 
    set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.50f}, Action{0.75f}, Action{1.00f}, Action{1.50f}, Action::ALL_IN}, 1, 1, 0);
  }

  // turn
  if(n_players > 2) {
    set_actions({Action::CHECK_CALL, Action{0.50f}, Action{1.00f}, Action::ALL_IN}, 2, 0, 0);
    set_actions({Action::FOLD, Action::CHECK_CALL, Action{1.00f}, Action::ALL_IN}, 2, 1, 0);
  }
  else {
    set_actions({Action::CHECK_CALL, Action{0.50f}, Action{1.00f}, Action{1.50f}, Action::ALL_IN}, 2, 0, 0);
    set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.50f}, Action{1.00f}, Action{1.50f}, Action::ALL_IN}, 2, 1, 0);
    if(stack_size < 10'000) add_action(Action{0.33f}, 2, 0, 0);
  }

  // river
  if(n_players > 2) {
    set_actions({Action::CHECK_CALL, Action{0.50f}, Action{1.00f}, Action::ALL_IN}, 3, 0, 0);  
    set_actions({Action::FOLD, Action::CHECK_CALL, Action{1.00f}, Action::ALL_IN}, 3, 1, 0);
  }
  else {
    set_actions({Action::CHECK_CALL, Action{0.50f}, Action{1.00f}, Action{1.50f}, Action::ALL_IN}, 3, 0, 0);  
    set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.50f}, Action{1.00f}, Action{1.50f}, Action::ALL_IN}, 3, 1, 0);
    if(stack_size < 7'500) add_action(Action{0.33f}, 3, 0, 0);
  }
}

BiasActionProfile::BiasActionProfile() {
  const std::vector bias_actions = {Action::BIAS_FOLD, Action::BIAS_CALL, Action::BIAS_RAISE, Action::BIAS_NONE};
  set_iso_actions(bias_actions);
  for(int round = 0; round <= 3; ++round) {
    set_actions(bias_actions, round, 0, 0);
  }
}

}
