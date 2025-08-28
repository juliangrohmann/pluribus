#include <random>
#include <pluribus/logging.hpp>
#include <pluribus/profiles.hpp>
#include <pluribus/rng.hpp>

#include "poker.hpp"

namespace pluribus {

std::vector<Action> action_vec(const std::initializer_list<float>& action_list, const bool can_fold = true) {
  std::vector<Action> actions;
  if(can_fold) actions.push_back(Action::FOLD);
  actions.push_back(Action::CHECK_CALL);
  for(float bet_size : action_list) actions.emplace_back(bet_size);
  actions.push_back(Action::ALL_IN);
  return actions;
}

std::vector<Action> action_range(const float start, const float end, const float step, const bool can_fold = true) {
  if(step <= 0.02f) Logger::error("Action range step is too small: " + std::to_string(step));
  if(start >= end) Logger::error("Invalid action range: [" + join_as_strs(std::vector{start, end, step}, ", ") + "]");
  std::vector<Action> actions;
  if(can_fold) actions.push_back(Action::FOLD);
  actions.push_back(Action::CHECK_CALL);
  for(int i = 0; start + static_cast<float>(i) * step + 0.01f < end; ++i) {
    actions.emplace_back(start + static_cast<float>(i) * step);
  }
  actions.emplace_back(end);
  actions.push_back(Action::ALL_IN);
  return actions;
}

std::vector<Action> single_size(const float bet_size, const bool can_fold = true) {
  if(bet_size <= 0.0f) Logger::error("Bet size is too small: " + std::to_string(bet_size));
  std::vector<Action> actions;
  if(can_fold) actions.push_back(Action::FOLD);
  actions.push_back(Action::CHECK_CALL);
  actions.emplace_back(bet_size);
  actions.push_back(Action::ALL_IN);
  return actions;
}

BiasActionProfile::BiasActionProfile() : ActionProfile{1} {
  const std::vector bias_actions = {Action::BIAS_FOLD, Action::BIAS_CALL, Action::BIAS_RAISE, Action::BIAS_NONE};
  set_iso_actions(bias_actions, 0, false);
  set_iso_actions(bias_actions, 0, true);
  for(int round = 0; round <= 3; ++round) {
    set_actions(bias_actions, round, 0, 0);
  }
}

HeadsUpBlueprintProfile::HeadsUpBlueprintProfile(const int stack_size) : ActionProfile{2} {
  // preflop RFI
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.75f}, Action::ALL_IN}, 0, 1, 0);
  set_iso_actions(action_range(1.00f, 2.00f, 0.50f), 0, false);
  set_iso_actions(action_range(1.00f, 2.00f, 0.50f), 0, true);

  // preflop 3-bet
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{1.00f}, Action{1.25f}, Action{1.50f}, Action{1.75f}, Action{2.00f}}, 0, 2, 0);
  if(stack_size < 10'000) add_action(Action{0.75f}, 0, 2, 0);

  // preflop 4-bet+
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.60f}, Action{0.70f}, Action{0.80f}, Action{0.90f}, Action{1.00f}, Action::ALL_IN}, 0, 3, 0);
  if(stack_size < 10'000) add_action(Action{0.50f}, 0, 3, 0);

  // flop
  set_actions({Action::CHECK_CALL, Action{0.16f}, Action{0.33f}, Action{0.50f}, Action{0.75f}, Action{1.00f}, Action::ALL_IN}, 1, 0, 0);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.50f}, Action{0.75f}, Action{1.00f}, Action{1.50f}, Action::ALL_IN}, 1, 1, 0);

  // turn
  set_actions({Action::CHECK_CALL, Action{0.50f}, Action{1.00f}, Action{1.50f}, Action::ALL_IN}, 2, 0, 0);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.50f}, Action{1.00f}, Action{1.50f}, Action::ALL_IN}, 2, 1, 0);
  if(stack_size < 10'000) add_action(Action{0.33f}, 2, 0, 0);

  // river
  set_actions({Action::CHECK_CALL, Action{0.50f}, Action{1.00f}, Action{1.50f}, Action::ALL_IN}, 3, 0, 0);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.50f}, Action{1.00f}, Action{1.50f}, Action::ALL_IN}, 3, 1, 0);
  if(stack_size < 7'500) add_action(Action{0.33f}, 3, 0, 0);
}

RingBlueprintProfile::RingBlueprintProfile(const int n_players) : ActionProfile{n_players} {
  // preflop RFI & isos
  for(int pos = 0; pos < 2; ++pos) set_actions(single_size(0.80f), 0, 1, pos);
  for(int pos = 2; pos < n_players; ++pos) set_actions(single_size(0.60f), 0, 1, pos);
  set_iso_actions(action_range(1.00, 2.00, 0.50), 0, false);
  set_iso_actions(action_vec({1.00, 1.50}), 0, true);

  // preflop 3-bet
  for(int pos = 0; pos < 2; ++pos) {
    set_actions(action_range(0.90, 1.90, 0.20), 0, 2, pos, false);
    set_actions(action_range(0.60, 1.80, 0.20), 0, 2, pos, true);
  }
  for(int pos = 2; pos < n_players; ++pos) {
    set_actions(action_range(0.90, 1.90, 0.20), 0, 2, pos, false);
    set_actions(action_range(0.60, 1.20, 0.20), 0, 2, pos, true);
  }

  // preflop 4-bet+
  set_actions(action_range(0.50, 1.20, 0.10), 0, 3, 0);

  // flop
  set_actions({Action::CHECK_CALL, Action{0.33f}, Action{0.50f}, Action{0.75f}, Action{1.00f}, Action::ALL_IN}, 1, 0, 0);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.50f}, Action{0.75f}, Action{1.00f}, Action::ALL_IN}, 1, 1, 0);

  // turn
  set_actions({Action::CHECK_CALL, Action{0.50f}, Action{1.00f}, Action::ALL_IN}, 2, 0, 0);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{1.00f}, Action::ALL_IN}, 2, 1, 0);

  // river
  set_actions({Action::CHECK_CALL, Action{0.50f}, Action{1.00f}, Action::ALL_IN}, 3, 0, 0);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{1.00f}, Action::ALL_IN}, 3, 1, 0);
}

void fill_from_profile(ActionProfile& to_profile, const ActionProfile& from_profile, const int max_round) {
  for(int pos = 0; pos < to_profile.n_players(); ++pos) {
    for(int in_position = 0; in_position <= 1; ++in_position) {
      const bool is_in_pos = static_cast<bool>(in_position);
      for(int r = 0; r <= max_round; ++r) {
        for(int bet_level = 0; bet_level <= 4; ++bet_level) {
          if(r == 0 && bet_level == 0) continue;
          to_profile.set_actions(from_profile.get_actions_from_raw(r, bet_level, pos, is_in_pos), r, bet_level, pos, is_in_pos);
        }
      }
      to_profile.set_iso_actions(from_profile.get_iso_actions(pos, is_in_pos), pos, is_in_pos);
    }
  }
}

WPTGoldRingBlueprintProfile::WPTGoldRingBlueprintProfile(const int n_players, const bool fine_grained) : ActionProfile{n_players} {
  // preflop RFI & isos
  for(int pos = 0; pos < 3; ++pos) set_actions(single_size(0.75), 0, 1, pos);
  for(int pos = 3; pos < n_players; ++pos) set_actions(single_size(0.42), 0, 1, pos);
  if(fine_grained) {
    set_iso_actions(action_range(1.00, 2.00, 0.50), 0, false);
    set_iso_actions(action_vec({1.00, 1.50}), 0, true);
  }
  else {
    set_iso_actions(single_size(1.50), 0, false);
    set_iso_actions(single_size(1.00), 0, true);
  }

  // preflop 3-bet
  if(fine_grained) {
    set_actions(action_range(0.70, 1.30, 0.15), 0, 2, 0, false);
    set_actions(action_range(0.70, 1.45, 0.15), 0, 2, 1, false);
    set_actions(action_range(0.50, 0.95, 0.15), 0, 2, 1, true);
    set_actions(action_range(0.85, 1.60, 0.15), 0, 2, 2, false);
    set_actions(action_range(0.50, 1.10, 0.15), 0, 2, 2, true);
    for(int pos = 3; pos < n_players; ++pos) {
      set_actions(action_range(0.70, 1.45, 0.15), 0, 2, pos, false);
      set_actions(action_range(0.55, 0.95, 0.10), 0, 2, pos, true);
    }
  }
  else {
    set_actions(action_range(0.90, 1.30, 0.20), 0, 2, 0, false);
    set_actions(action_range(1.00, 1.40, 0.20), 0, 2, 1, false);
    set_actions(action_range(0.50, 0.90, 0.20), 0, 2, 1, true);
    set_actions(action_range(1.00, 1.40, 0.20), 0, 2, 2, false);
    set_actions(action_range(0.60, 1.00, 0.20), 0, 2, 2, true);
    for(int pos = 3; pos < n_players; ++pos) {
      set_actions(action_range(0.90, 1.30, 0.20), 0, 2, pos, false);
      set_actions(action_range(0.55, 0.85, 0.15), 0, 2, pos, true);
    }
  }

  // preflop 4-bet
  set_actions(action_range(0.40, 1.00, 0.10), 0, 3, 0);

  // flop
  if(fine_grained) {
    set_actions({Action::CHECK_CALL, Action{0.33f}, Action{0.50f}, Action{0.75f}, Action{1.00f}, Action::ALL_IN}, 1, 0, 0);
    set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.50f}, Action{0.75f}, Action{1.00f}, Action::ALL_IN}, 1, 1, 0);
  }
  else {
    set_actions({Action::CHECK_CALL, Action{0.33f}, Action{0.67f}, Action{1.00f}, Action::ALL_IN}, 1, 0, 0);
    set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.75f}, Action::ALL_IN}, 1, 1, 0);
  }
  // turn
  set_actions({Action::CHECK_CALL, Action{0.50f}, Action{1.00f}, Action::ALL_IN}, 2, 0, 0);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{1.00f}, Action::ALL_IN}, 2, 1, 0);

  // river
  set_actions({Action::CHECK_CALL, Action{0.50f}, Action{1.00f}, Action::ALL_IN}, 3, 0, 0);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{1.00f}, Action::ALL_IN}, 3, 1, 0);
}

HeadsUpLiveProfile::HeadsUpLiveProfile() : ActionProfile{2} {
  fill_from_profile(*this, HeadsUpBlueprintProfile{10'000}, 1);
  for(int pos = 0; pos < n_players(); ++pos) {
    for(int in_position = 0; in_position <= 1; ++in_position) {
      const bool is_in_pos = static_cast<bool>(in_position);
      set_actions({Action::CHECK_CALL, Action{0.33f}, Action{0.50f}, Action{0.75f}, Action{1.00f}, Action{1.25f}, Action{1.50f}, Action::ALL_IN},
          2, 0, pos, is_in_pos); // turn bet
      set_actions({Action::CHECK_CALL, Action{0.33f}, Action{0.50f}, Action{0.75f}, Action{1.00f}, Action{1.50f}, Action{2.00f}, Action::ALL_IN},
          3, 0, pos, is_in_pos); // river bet
      for(int bet_level = 1; bet_level <= 4; ++bet_level) {
        set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.50f}, Action{0.75f}, Action{1.00f}, Action::ALL_IN}, 2, 1, pos, is_in_pos); // turn raise
        set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.50f}, Action{0.75f}, Action{1.00f}, Action::ALL_IN}, 3, 1, pos, is_in_pos); // river raise
      }
    }
  }
}

RingLiveProfile::RingLiveProfile(const int n_players, const int round) : ActionProfile{n_players} {
  // TODO: unify blueprint/live profile
  fill_from_profile(*this, RingBlueprintProfile{n_players}, 3);
  for(int pos = 0; pos < n_players; ++pos) {
    for(int in_position = 0; in_position <= 1; ++in_position) {
      const bool is_in_pos = static_cast<bool>(in_position);
      if(round >= 2) {
        set_actions({Action::CHECK_CALL, Action{0.33f}, Action{0.50f}, Action{0.75f}, Action{1.00f}, Action{1.25f}, Action{1.50f}, Action::ALL_IN},
            2, 0, pos, is_in_pos); // turn bet
        for(int bet_level = 1; bet_level <= 4; ++bet_level) { // turn raise
          set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.50f}, Action{0.75f}, Action{1.00f}, Action::ALL_IN}, 2, 1, pos, is_in_pos);
        }
      }
      if(round >= 3) {
        set_actions({Action::CHECK_CALL, Action{0.33f}, Action{0.50f}, Action{0.75f}, Action{1.00f}, Action{1.50f}, Action{2.00f}, Action::ALL_IN},
            3, 0, pos, is_in_pos); // river bet
        for(int bet_level = 1; bet_level <= 4; ++bet_level) { // river raise
          set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.50f}, Action{0.75f}, Action{1.00f}, Action::ALL_IN}, 3, 1, pos, is_in_pos);
        }
      }
    }
  }
}

}
