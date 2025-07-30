#pragma once

#include <initializer_list>
#include <memory>
#include <string>
#include <boost/dynamic_bitset.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <hand_isomorphism/hand_index.h>

namespace pluribus {

class Action {
public:
  explicit Action(const float bet_type = -3.0f) : _bet_type{bet_type} {}
  
  float get_bet_type() const { return _bet_type; }
  std::string to_string() const;

  bool operator==(const Action& other) const { return _bet_type == other._bet_type; }

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

class PokerState;

using ProfileStorage = std::array<std::vector<std::vector<std::vector<std::vector<Action>>>>, 4>;

class ActionProfile {
public:
  explicit ActionProfile(const int n_players = -1) : _n_players{n_players} {}
  void set_actions(const std::vector<Action>& actions, int round, int bet_level, int pos, bool in_position = false);
  void set_iso_actions(const std::vector<Action>& actions, int pos);
  void add_action(Action action, int round, int bet_level, int pos, bool in_position = false);
  void add_iso_action(Action action, int pos);
  const std::vector<Action>& get_actions_from_raw(int round, int bet_level, int pos, bool in_position) const;
  const std::vector<Action>& get_iso_actions(int pos) const;
  const std::vector<Action>& get_actions(const PokerState& state) const;
  const ProfileStorage& get_raw_profile() const { return _profile; }
  int n_bet_levels(const int round) const { return static_cast<int>(_profile[round].size()); }
  std::unordered_set<Action> all_actions() const;
  int max_actions() const;
  int max_bet_level() const;
  int n_players() const { return _n_players; }
  std::string to_string() const;

  bool operator==(const ActionProfile&) const = default;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_profile, _iso_actions, _n_players);
  }

protected:
  void set_raw_profile(const ProfileStorage& raw_profile) { _profile = raw_profile; }

private:
  void grow_to_fit(int round, int bet_level, int pos, bool in_position);
  void sort(int round, int bet_level, int pos, bool in_position);

  ProfileStorage _profile;
  std::vector<std::vector<Action>> _iso_actions;
  int _n_players;
};

class CombinedActionProfile : public ActionProfile {
public:
  CombinedActionProfile(int hero_pos, const ActionProfile& hero_profile, const ActionProfile& villain_profile, int max_round = 3);
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
    for(const pluribus::Action& a : ah.get_history()) {
      constexpr std::hash<pluribus::Action> action_hasher;
      boost::hash_combine(seed, action_hasher(a));
    }
    return seed;
  }
};

}
