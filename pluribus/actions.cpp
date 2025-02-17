#include <iostream>
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

size_t ActionHistory::size() const {
  return _end_idx / action_bits;
}

std::string ActionHistory::to_string() const {
  std::string str = "";
  for(int i = 0; i < size(); ++i) {
    str += action_to_str.at(get(i)) + (i == size() - 1 ? "" : " -> ");
  }
  return str;
}

}
