#pragma once

#include <initializer_list>
#include <hand_isomorphism/hand_index.h>
#include <boost/dynamic_bitset.hpp>

namespace pluribus {

const int action_bits = 4;
enum class Action : uint8_t {
  UNDEFINED = 0,
  FOLD = 1,
  CHECK_CALL = 2,
  BET_33 = 3,
  BET_50 = 4,
  BET_75 = 5,
  BET_100 = 6,
  BET_150 = 7,
  PREFLOP_2_BET = 8,
  PREFLOP_3_BET = 9,
  PREFLOP_4_BET = 10,
  PREFLOP_5_BET = 11,
  POSTFLOP_2_BET = 12,
  POSTFLOP_3_BET = 13,
  ALL_IN = 14
};

class ActionHistory {
public:
  ActionHistory() {};
  ActionHistory(std::initializer_list<Action> list);
  void push_back(Action action);
  Action get(int idx) const;
  size_t size() const;
  const boost::dynamic_bitset<unsigned long>& data() const;
private:
  boost::dynamic_bitset<unsigned long> _data;
  size_t _end_idx = 0;
};

}

namespace std {

template <>
struct hash<pluribus::ActionHistory> {
  std::size_t operator()(const pluribus::ActionHistory& ah) const {
    return boost::hash_value(ah.data());
  }
};

}
