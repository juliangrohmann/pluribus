#pragma once

#include <string>
#include <unordered_map>
#include <initializer_list>
#include <memory>
#include <cereal/cereal.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/archives/json.hpp>
#include <boost/dynamic_bitset.hpp>
#include <hand_isomorphism/hand_index.h>

namespace pluribus {

class Action {
public:
  Action(float bet_type = -3.0f) : _bet_type{bet_type} {}
  
  float get_bet_type() const { return _bet_type; };
  std::string to_string() const;

  bool operator==(const Action& other) const { return _bet_type == other._bet_type; }

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_bet_type);
  }

  static Action UNDEFINED;
  static Action ALL_IN;
  static Action FOLD;
  static Action CHECK_CALL;

private:
  float _bet_type;
};

class ActionHistory {
public:
  ActionHistory() = default;
  ActionHistory(std::initializer_list<Action> actions) : _history{actions} {}

  const std::vector<Action> get_history() const { return _history; }
  void push_back(const Action& action) { _history.push_back(action); }
  const Action& get(int i ) { return _history[i]; }
  size_t size() const { return _history.size(); }
  std::string to_string() const;

  bool operator==(const ActionHistory& other) const { return _history == other._history; }

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_history);
  }

private:
  std::vector<Action> _history;
};

class ActionProfile {
public:
  void set_actions(const std::vector<Action>& actions, int round, int bet_level);
  const std::vector<Action>& get_actions(int round, int bet_level) const;
  void add_action(const Action& action, int round, int bet_level);
  std::string to_string() const;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_profile);
  }

private:
  std::array<std::vector<std::vector<Action>>, 4> _profile;
};

class BlueprintActionProfile : public ActionProfile {
public:
  BlueprintActionProfile();
};

using HistoryMap = std::unordered_map<ActionHistory, int>;

class HistoryIndexer {
public:
  void initialize(int n_players, int n_chips, int ante);
  int index(const ActionHistory& history, int n_players, int n_chips, int ante);
  size_t size(int n_players, int n_chips, int ante);

  static HistoryIndexer* get_instance() {
    if(!_instance) {
      _instance = std::unique_ptr<HistoryIndexer>(new HistoryIndexer());
    }
    return _instance.get();
  }

  HistoryIndexer(const HistoryIndexer&) = delete;
  HistoryIndexer& operator==(const HistoryIndexer&) = delete;
private:
  HistoryIndexer() {}

  std::unordered_map<std::string, HistoryMap> _history_map;

  static std::unique_ptr<HistoryIndexer> _instance;
};

void build_history_map(int n_players, int n_chips, int ante);

}

namespace std {

template <>
struct hash<pluribus::Action> {
  std::size_t operator()(const pluribus::Action& a) const {
    return std::hash<float>{}(a.get_bet_type());
  }
};

template <>
struct hash<pluribus::ActionHistory> {
  std::size_t operator()(const pluribus::ActionHistory& ah) const {
    size_t seed = 0;
    std::hash<pluribus::Action> action_hasher;    
    for(const pluribus::Action& a : ah.get_history()) {
      boost::hash_combine(seed, action_hasher(a));
    }
    return seed;
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
