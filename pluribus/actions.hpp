#pragma once

#include <string>
#include <unordered_map>
#include <initializer_list>
#include <cereal/cereal.hpp>
#include <cereal/types/string.hpp>
#include <cereal/archives/json.hpp>
#include <boost/dynamic_bitset.hpp>
#include <hand_isomorphism/hand_index.h>

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

const std::unordered_map<Action, std::string> action_to_str {
  {Action::UNDEFINED, "Undefined"},
  {Action::FOLD, "Fold"},
  {Action::CHECK_CALL, "Check/Call"},
  {Action::BET_33, "Bet 33%"},
  {Action::BET_50, "Bet 50%"},
  {Action::BET_75, "Bet 75%"},
  {Action::BET_100, "Bet 100%"},
  {Action::BET_150, "Bet 150%"},
  {Action::PREFLOP_2_BET, "Preflop 2-bet"},
  {Action::PREFLOP_3_BET, "Preflop 3-bet"},
  {Action::PREFLOP_4_BET, "Preflop 4-bet"},
  {Action::PREFLOP_5_BET, "Preflop 5-bet"},
  {Action::POSTFLOP_2_BET, "Postflop 2-bet"},
  {Action::POSTFLOP_3_BET, "Postflop 3-bet"},
  {Action::ALL_IN, "All-in"}
};

class ActionHistory {
public:
  ActionHistory() {};
  ActionHistory(const ActionHistory&) = default;
  ActionHistory(std::initializer_list<Action> list);
  bool operator==(const ActionHistory& other) const;
  void push_back(Action action);
  Action get(int idx) const;
  uint16_t size() const;
  inline const boost::dynamic_bitset<unsigned long>& data() const { return _data; }
  std::string to_string() const;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_data, _end_idx);
  }
private:
  boost::dynamic_bitset<unsigned long> _data;
  uint16_t _end_idx = 0;
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

namespace cereal {

template<class Archive, typename Block, typename Allocator>
void save(Archive& ar, const boost::dynamic_bitset<Block, Allocator>& bits) {
  size_t size = bits.size();
  ar(size);
  std::string bitStr;
  boost::to_string(bits, bitStr);
  ar(bitStr);
}

template<class Archive, typename Block, typename Allocator>
void load(Archive& ar, boost::dynamic_bitset<Block, Allocator>& bits) {
  size_t size;
  ar(size);
  std::string bitStr;
  ar(bitStr);
  bits = boost::dynamic_bitset<Block, Allocator>(bitStr);
}

// template<class Archive, typename Block, typename Allocator>
// inline void serialize(Archive& ar, boost::dynamic_bitset<Block, Allocator>& bits) {
//   cereal::split_free(ar, bits);
// }

}
