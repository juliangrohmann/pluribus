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

std::string Action::to_string() const {
  if(_bet_type == -3.0f) return "Undefined";
  if(_bet_type == -2.0f) return "All-in";
  if(_bet_type == -1.0f) return "Fold";
  if(_bet_type == 0.0f) return "Check/Call";
  std::ostringstream oss; 
  oss << "Bet " << std::setprecision(0) << std::fixed << _bet_type * 100.0f << "%";
  return oss.str();
}

Action Action::UNDEFINED{-3.0f};
Action Action::ALL_IN{-2.0f};
Action Action::FOLD{-1.0f};
Action Action::CHECK_CALL{0.0f};

ActionHistory ActionHistory::slice(int start, int end) const { 
  return ActionHistory{std::vector<Action>{_history.begin() + start, end != -1 ? _history.begin() + end : _history.end()}}; 
}

std::string ActionHistory::to_string() const {
  std::string str = "";
  for(int i = 0; i < _history.size(); ++i) {
    str += _history[i].to_string() + (i == _history.size() - 1 ? "" : ", ");
  }
  return str;
}

void ActionProfile::set_actions(const std::vector<Action>& actions, int round, int bet_level) {
  if(bet_level >= _profile[round].size()) _profile[round].resize(bet_level + 1);
  _profile[round][bet_level] = actions;
}

const std::vector<Action>& ActionProfile::get_actions(int round, int bet_level) const { 
  int max_level = _profile[round].size() - 1;
  return _profile[round][std::min(bet_level, max_level)]; 
}

void ActionProfile::add_action(const Action& action, int round, int bet_level) {
  if(bet_level < _profile[round].size()) _profile[round].resize(bet_level);
  _profile[round][bet_level].push_back(action);
}

int ActionProfile::max_actions() const {
  int ret;
  for(auto& round : _profile) {
    for(auto& level : round) {
      ret = std::max(static_cast<int>(level.size()), ret);
    }
  }
  return ret;
}

std::string ActionProfile::to_string() const {
  std::ostringstream oss;
  for(int round = 0; round < 4; ++round) {
    oss << round_to_str(round) << " action profile:\n";
    for(int bet_level = 0; bet_level < _profile[round].size(); ++bet_level) {
      oss << "\tBet level " << bet_level << ":\n";
      for(Action a : _profile[round][bet_level]) {
        oss << "\t\t" << a.to_string() << "\n";
      }
    }
  }
  return oss.str();
}

BlueprintActionProfile::BlueprintActionProfile() {
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.60f}, Action{0.80f}, Action::ALL_IN}, 0, 1);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.60f}, Action{0.80f}, Action{1.00f}, Action{1.20f}, Action::ALL_IN}, 0, 2);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.60f}, Action{0.80f}, Action{1.00f}, Action::ALL_IN}, 0, 3);

  set_actions({Action::CHECK_CALL, Action{0.16}, Action{0.33f}, Action{0.50f}, Action{0.75f}, Action{1.00f}, Action::ALL_IN}, 1, 0);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.50f}, Action{0.75f}, Action{1.00f}, Action::ALL_IN}, 1, 1);

  set_actions({Action::CHECK_CALL, Action{0.50f}, Action{1.00f}, Action::ALL_IN}, 2, 0);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{1.00f}, Action::ALL_IN}, 2, 1);

  set_actions({Action::CHECK_CALL, Action{0.50f}, Action{1.00f}, Action::ALL_IN}, 3, 0);  
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{1.00f}, Action::ALL_IN}, 3, 1);
}

}
