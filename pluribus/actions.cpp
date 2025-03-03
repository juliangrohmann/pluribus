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
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{2.00f}, Action::ALL_IN}, 0, 1);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{2.00f}, Action::ALL_IN}, 0, 2);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{1.66f}, Action::ALL_IN}, 0, 3);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{1.50f}, Action::ALL_IN}, 0, 4);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action::ALL_IN}, 0, 5);

  set_actions({Action::CHECK_CALL, Action{0.33f}, Action{0.75f}, Action::ALL_IN}, 1, 0);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{1.50f}, Action::ALL_IN}, 1, 1);

  set_actions({Action::CHECK_CALL, Action{0.50f}, Action{1.00f}, Action::ALL_IN}, 2, 0);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{1.50f}, Action::ALL_IN}, 2, 1);

  set_actions({Action::CHECK_CALL, Action{0.50f}, Action{1.00f}, Action::ALL_IN}, 3, 0);  
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{1.50f}, Action::ALL_IN}, 3, 1);
}

std::string history_map_filename(int n_players, int n_chips, int ante) {
  return "history_index_p" + std::to_string(n_players) + "_" + std::to_string(n_chips / 100) + "bb_" + std::to_string(ante) + "ante.bin";
}

std::unique_ptr<HistoryIndexer> HistoryIndexer::_instance = nullptr;

void HistoryIndexer::initialize(int n_players, int n_chips, int ante) {
  std::string fn = history_map_filename(n_players, n_chips, ante);
  if(_history_map.find(fn) == _history_map.end()) {
    std::cout << "Initializing history map (n_players=" << n_players << ", n_chips=" << n_chips << ", ante=" << ante << ")... " << std::flush;
    _history_map[fn] = cereal_load<HistoryMap>(fn);
    std::cout << "Success.\n";
  }
}

int HistoryIndexer::index(const ActionHistory& history, int n_players, int n_chips, int ante) {
  std::string fn = history_map_filename(n_players, n_chips, ante);
  auto it = _history_map.find(fn);
  if(it == _history_map.end()) {
    initialize(n_players, n_chips, ante);
    return _history_map.at(fn).at(history);
  }
  return it->second.at(history);
}

size_t HistoryIndexer::size(int n_players, int n_chips, int ante) {
  std::string fn = history_map_filename(n_players, n_chips, ante);
  auto it = _history_map.find(fn);
  if(it != _history_map.end()) {
    return it->second.size();
  }
  else {
    throw std::runtime_error("HistoryIndexer::size --- HistoryIndexer is uninitialized.");
  }
}

void collect_histories(const PokerState& state, std::vector<ActionHistory>& histories, const ActionProfile& action_profile) {
  if(state.is_terminal()) return;
  histories.push_back(state.get_action_history());
  for(Action a : valid_actions(state, action_profile)) {
    collect_histories(state.apply(a), histories, action_profile);
  }
}

void build_history_map(int n_players, int n_chips, int ante, const ActionProfile& action_profile) {
  std::cout << "Building history map: n_players=" << n_players << ", n_chips=" << n_chips << ", ante=" << ante << "\n";
  std::vector<ActionHistory> histories;
  PokerState state{n_players, n_chips, ante};
  collect_histories(state, histories, action_profile);
  HistoryMap history_map;
  for(long i = 0; i < histories.size(); ++i) {
    history_map[histories[i]] = i;
  }
  std::cout << "n=" << histories.size() << "\n";
  cereal_save(history_map, history_map_filename(n_players, n_chips, ante));
}

}
