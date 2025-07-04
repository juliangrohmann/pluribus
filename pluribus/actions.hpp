#pragma once

#include <string>
#include <initializer_list>
#include <memory>
#include <cereal/cereal.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <boost/dynamic_bitset.hpp>
#include <hand_isomorphism/hand_index.h>

namespace pluribus {

class Action {
public:
  explicit Action(const float bet_type = -3.0f) : _bet_type{bet_type} {}
  
  float get_bet_type() const { return _bet_type; }
  std::string to_string() const;

  bool operator==(const Action& other) const = default;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_bet_type);
  }

  static const Action UNDEFINED;
  static const Action BIAS_DUMMY;
  static const Action BIAS_FOLD;
  static const Action BIAS_CALL;
  static const Action BIAS_RAISE;
  static const Action BIAS_NONE;
  static const Action ALL_IN;
  static const Action FOLD;
  static const Action CHECK_CALL;

private:
  float _bet_type;
};

bool is_bias(Action a);

class ActionHistory {
public:
  ActionHistory() = default;

  explicit ActionHistory(const std::vector<Action>& actions) : _history{actions} {}
  ActionHistory(const std::initializer_list<Action>& actions) : _history{actions} {}

  void push_back(const Action& action) { _history.push_back(action); }
  const std::vector<Action>& get_history() const { return _history; }
  const Action& get(const int i) const { return _history[i]; }
  size_t size() const { return _history.size(); }
  std::string to_string() const;
  ActionHistory slice(int start, int end = -1) const;
  bool is_consistent(const ActionHistory& other) const;

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
  void set_actions(const std::vector<Action>& actions, int round, int bet_level, int pos);
  void set_iso_actions(const std::vector<Action>& actions) { _iso_actions = actions; }
  void add_action(const Action& action, int round, int bet_level, int pos);
  const std::vector<Action>& get_actions(int round, int bet_level, int pos, int pot) const;
  int n_bet_levels(const int round) const { return static_cast<int>(_profile[round].size()); }
  std::unordered_set<Action> all_actions() const;
  int max_actions() const;
  std::string to_string() const;

  bool operator==(const ActionProfile&) const = default;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_profile, _iso_actions);
  }

private:
  void grow_to_fit(int round, int bet_level, int pos);
  void sort(int round, int bet_level, int pos);

  std::array<std::vector<std::vector<std::vector<Action>>>, 4> _profile;
  std::vector<Action> _iso_actions;
};

class BlueprintActionProfile : public ActionProfile {
public:
  explicit BlueprintActionProfile(int n_players, int stack_size);
};

class BiasActionProfile : public ActionProfile {
public:
  BiasActionProfile();
};

}

namespace std {

template <>
struct hash<pluribus::Action> {
  std::size_t operator()(const pluribus::Action& a) const noexcept {
    return std::hash<float>{}(a.get_bet_type());
  }
};

template <>
struct hash<pluribus::ActionHistory> {
  std::size_t operator()(const pluribus::ActionHistory& ah) const noexcept {
    size_t seed = 0;
    constexpr std::hash<pluribus::Action> action_hasher;
    for(const pluribus::Action& a : ah.get_history()) {
      boost::hash_combine(seed, action_hasher(a));
    }
    return seed;
  }
};

}
