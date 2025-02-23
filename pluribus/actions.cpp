#include <iostream>
#include <vector>
#include <cereal/cereal.hpp>
#include <pluribus/util.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/actions.hpp>

namespace pluribus {

ActionHistory::ActionHistory(std::initializer_list<Action> list) {
  for(Action action : list) {
    push_back(action);
  }
};

bool ActionHistory::operator==(const ActionHistory& other) const {
  return _data == other._data;
}

void ActionHistory::push_back(Action action) {
  if(_end_idx >= _data.size()) {
    int block_bits = std::numeric_limits<decltype(_data)::block_type>::digits;
    _data.resize(_data.size() + block_bits);
  }
  assert(_end_idx % action_bits == 0 && "ActionHistory end index is misaligned.");
  assert(_end_idx + action_bits <= _data.size() && "Action counter is out of range.");
  uint8_t action_id = static_cast<uint8_t>(action);
  for(int i = 0; i < action_bits; ++i) {
    _data[_end_idx + i] = action_id & (1 << i);
  }
  _end_idx += action_bits;
}

Action ActionHistory::get(int idx) const {
  uint8_t action_id = 0;
  for(int i = 0; i < action_bits; ++i) {
    action_id |= _data[action_bits * idx + i] << i;
  }
  return static_cast<Action>(action_id);
}

uint16_t ActionHistory::size() const {
  return _end_idx / action_bits;
}

std::string ActionHistory::to_string() const {
  std::string str = "";
  for(int i = 0; i < size(); ++i) {
    str += action_to_str.at(get(i)) + (i == size() - 1 ? "" : " -> ");
  }
  return str;
}

std::string history_map_filename(int n_players, int n_chips, int ante) {
  return "history_index_p" + std::to_string(n_players) + "_" + std::to_string(n_chips / 100) + "bb_" + std::to_string(ante) + "ante.bin";
}

std::unordered_map<std::string, HistoryMap> HistoryIndexer::_history_map;

void HistoryIndexer::initialize(int n_players, int n_chips, int ante) {
  std::string fn = history_map_filename(n_players, n_chips, ante);
  if(_history_map.find(fn) == _history_map.end()) {
    std::cout << "Initializing history map: n_players=" << n_players << ", n_chips=" << n_chips << ", ante=" << ante << "\n";
    _history_map[fn] = cereal_load<HistoryMap>(fn);
  }
  else {
    std::cout << "Already initialized: n_players=" << n_players << ", n_chips=" << n_chips << ", ante=" << ante << "\n";
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

void collect_histories(const PokerState& state, std::vector<ActionHistory>& histories) {
  if(state.is_terminal()) return;
  histories.push_back(state.get_action_history());
  for(Action a : valid_actions(state)) {
    collect_histories(state.apply(a), histories);
  }
}

void build_history_map(int n_players, int n_chips, int ante) {
  std::cout << "Building history map: n_players=" << n_players << ", n_chips=" << n_chips << ", ante=" << ante << "\n";
  std::vector<ActionHistory> histories;
  PokerState state{n_players, n_chips, ante};
  collect_histories(state, histories);
  HistoryMap history_map;
  for(long i = 0; i < histories.size(); ++i) {
    history_map[histories[i]] = i;
  }
  std::cout << "n=" << histories.size() << "\n";
  cereal_save(history_map, history_map_filename(n_players, n_chips, ante));
}

}
