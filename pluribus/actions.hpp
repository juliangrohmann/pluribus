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

  bool operator==(const Action& other) const = default;

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
  ActionHistory(std::vector<Action> actions) : _history{actions} {}
  ActionHistory(std::initializer_list<Action> actions) : _history{actions} {}

  const std::vector<Action> get_history() const { return _history; }
  void push_back(const Action& action) { _history.push_back(action); }
  const Action& get(int i) const { return _history[i]; }
  size_t size() const { return _history.size(); }
  std::string to_string() const;
  ActionHistory slice(int start, int end = -1) const; 

  bool operator==(const ActionHistory& other) const = default;

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
  int max_actions() const;
  std::string to_string() const;

  bool operator==(const ActionProfile&) const = default;

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
