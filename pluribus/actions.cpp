#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <limits>
#include <cereal/cereal.hpp>
#include <pluribus/util.hpp>
#include <pluribus/debug.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/actions.hpp>
#include <sys/stat.h>

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

template <class T>
void grow_by_copy(std::vector<T>& vec, int size) {
  if(vec.size() == 0) vec.resize(1);
  while(vec.size() < size) vec.push_back(vec[vec.size() - 1]);
}

void ActionProfile::grow_to_fit(const int round, const int bet_level, const int pos, const bool in_position) {
  grow_by_copy(_profile[round], bet_level + 1);
  grow_by_copy(_profile[round][bet_level], pos + 1);
  grow_by_copy(_profile[round][bet_level][pos], in_position + 1);
}

float sort_key(const Action a) {
  return a == Action::ALL_IN ? std::numeric_limits<float>::max() : a.get_bet_type();
}

void ActionProfile::sort(const int round, const int bet_level, const int pos, const bool in_position) {
  std::ranges::sort(_profile[round][bet_level][pos][static_cast<int>(in_position)], std::ranges::less{}, &sort_key);
}

void ActionProfile::set_actions(const std::vector<Action>& actions, const int round, const int bet_level, const int pos, const bool in_position) {
  grow_to_fit(round, bet_level, pos, in_position);
  _profile[round][bet_level][pos][in_position] = actions;
  sort(round, bet_level, pos, in_position);
}

void ActionProfile::add_action(const Action& action, const int round, const int bet_level, const int pos, const bool in_position) {
  grow_to_fit(round, bet_level, pos, in_position);
  _profile[round][bet_level][pos][in_position].push_back(action);
  sort(round, bet_level, pos, in_position);
}

const std::vector<Action>& ActionProfile::get_actions(const PokerState& state) const {
  if(state.get_round() == 0 && state.get_bet_level() == 1 && state.vpip_players() > 0) return _iso_actions;
  const int level_idx = std::min(static_cast<int>(state.get_bet_level()), static_cast<int>(_profile[state.get_round()].size()) - 1);
  const int pos_idx = std::min(static_cast<int>(state.get_active()), static_cast<int>(_profile[state.get_round()][level_idx].size()) - 1);
  const auto& ip_vec = _profile[state.get_round()][level_idx][pos_idx];
  if(ip_vec.size() == 1) return ip_vec[0];
  return ip_vec[static_cast<int>(state.is_in_position(state.get_active()))];
}

std::unordered_set<Action> ActionProfile::all_actions() const {
  std::unordered_set<Action> actions;
  for(auto& round : _profile) {
    for(auto& level : round) {
      for(auto& pos : level) {
        for(auto& ip : pos) {
          for(Action a : ip) {
            actions.insert(a);
          }
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
        for(auto& ip : pos) {
          ret = std::max(static_cast<int>(ip.size()), ret);
        }
      }
    }
  }
  return std::max(ret, static_cast<int>(_iso_actions.size()));
}

std::string actions_to_str(const std::vector<Action>& actions) {
  std::ostringstream oss;
  for(Action a : actions) {
    oss << a.to_string() << "  ";
  }
  return oss.str();
}

std::string ActionProfile::to_string() const {
  std::ostringstream oss;
  oss << "Iso actions: " << actions_to_str(_iso_actions) << "\n";
  for(int round = 0; round < 4; ++round) {
    oss << round_to_str(round) << " action profile:\n";
    for(int bet_level = 0; bet_level < _profile[round].size(); ++bet_level) {
      oss << "\tBet level " << bet_level << ":\n";
      for(int pos = 0; pos < _profile[round][bet_level].size(); ++pos) {
        oss << "\t" << "Position " << pos << ":";
        for(int ip = 0; ip < _profile[round][bet_level][pos].size(); ++ip) {
          oss << (ip == 0 ? "\t\tOOP:  " : "\t\tIP:  ");
          for(Action a : _profile[round][bet_level][pos][ip]) {
            oss << a.to_string() << "  ";
          }
          oss << "\n";
        }
      }
    }
  }
  return oss.str();
}

}
