#include <random>
#include <pluribus/logging.hpp>
#include <pluribus/profiles.hpp>
#include <pluribus/rng.hpp>

#include "poker.hpp"

namespace pluribus {

BiasActionProfile::BiasActionProfile() : ActionProfile{1} {
  const std::vector bias_actions = {Action::BIAS_FOLD, Action::BIAS_CALL, Action::BIAS_RAISE, Action::BIAS_NONE};
  set_iso_actions(bias_actions, 0);
  for(int round = 0; round <= 3; ++round) {
    set_actions(bias_actions, round, 0, 0);
  }
}

HeadsUpBlueprintProfile::HeadsUpBlueprintProfile(const int stack_size) : ActionProfile{2} {
  // preflop RFI
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.75f}, Action::ALL_IN}, 0, 1, 0);
  set_iso_actions({Action::FOLD, Action::CHECK_CALL, Action{1.00f}, Action{2.00f}, Action::ALL_IN}, 0);

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

HeadsUpSimpleProfile::HeadsUpSimpleProfile(const int stack_size) : ActionProfile{2} {
  // preflop RFI
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.75f}, Action::ALL_IN}, 0, 1, 0);
  set_iso_actions({Action::FOLD, Action::CHECK_CALL, Action{1.00f}, Action::ALL_IN}, 0);

  // preflop 3-bet
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{1.50f}}, 0, 2, 0);

  // preflop 4-bet+
  set_actions({Action::FOLD, Action::CHECK_CALL, Action::ALL_IN}, 0, 3, 0);
  if(stack_size < 10'000) add_action(Action{0.55f}, 0, 3, 0);
  else if(stack_size < 15'000) add_action(Action{0.70f}, 0, 3, 0);
  else if(stack_size < 20'000) add_action(Action{0.80f}, 0, 3, 0);
  else add_action(Action{0.85f}, 0, 3, 0);

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

RingBlueprintProfile::RingBlueprintProfile(const int n_players, const int stack_size) : ActionProfile{n_players} {
  // preflop RFI & isos
  for(int pos = 2; pos < n_players - 2; ++pos) {
    set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.60f}, Action::ALL_IN}, 0, 1, pos);
  }
  if(n_players > 3) set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.60f}, Action::ALL_IN}, 0, 1, n_players - 2);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.60f}, Action::ALL_IN}, 0, 1, n_players - 1);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.80f}, Action::ALL_IN}, 0, 1, 0);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.80f}, Action::ALL_IN}, 0, 1, 1);
  set_iso_actions({Action::FOLD, Action::CHECK_CALL, Action{0.85f}, Action{1.00f}, Action{1.25}, Action{1.50}, Action{2.00f}, Action::ALL_IN}, 0);

  // preflop 3-bet
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{1.10f}, Action{1.40f}, Action{1.70f}, Action::ALL_IN}, 0, 2, 0);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{1.00f}, Action{1.30f}, Action{1.60f}, Action::ALL_IN}, 0, 2, 1);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.80f}, Action{1.00f}, Action{1.20f}, Action::ALL_IN}, 0, 2, 2);

  // preflop 4-bet+
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.60f}, Action{0.70f}, Action{0.80f}, Action{0.90f}, Action{1.00f}, Action::ALL_IN}, 0, 3, 0);
  if(stack_size < 10'000) add_action(Action{0.50f}, 0, 3, 0);

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

RingSimpleProfile::RingSimpleProfile(const int n_players, const int stack_size) : ActionProfile{n_players} {
  // preflop RFI & isos
  for(int pos = 2; pos < n_players - 2; ++pos) {
    set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.60f}, Action::ALL_IN}, 0, 1, pos);
  }
  if(n_players > 3) set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.60f}, Action::ALL_IN}, 0, 1, n_players - 2);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.60f}, Action::ALL_IN}, 0, 1, n_players - 1);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.80f}, Action::ALL_IN}, 0, 1, 0);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.80f}, Action::ALL_IN}, 0, 1, 1);
  set_iso_actions({Action::FOLD, Action::CHECK_CALL, Action{0.85f}, Action::ALL_IN}, 0);

  // preflop 3-bet
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{1.40f}, Action::ALL_IN}, 0, 2, 0);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{1.30f}, Action::ALL_IN}, 0, 2, 1);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.80f}, Action::ALL_IN}, 0, 2, 2);

  // preflop 4-bet+
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.70f}, Action::ALL_IN}, 0, 3, 0, false);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.60f}, Action::ALL_IN}, 0, 3, 0, true);

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

WPTGoldRingBlueprintProfile::WPTGoldRingBlueprintProfile(const int n_players) : ActionProfile{n_players} {
  // preflop RFI & isos
  for(int pos = 0; pos < 3; ++pos) {
    set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.85f}, Action::ALL_IN}, 0, 1, pos);
  }
  for(int pos = 3; pos < n_players; ++pos) {
    set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.52f}, Action::ALL_IN}, 0, 1, pos);
  }
  set_iso_actions({Action::FOLD, Action::CHECK_CALL, Action{0.70f}, Action{1.00f}, Action{1.50f}, Action{2.00f}, Action::ALL_IN}, 0);
  if(n_players <= 6) add_iso_action(Action{0.52f}, 0);

  // preflop 3-bet
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.83f}, Action{1.00f}, Action{1.17f}, Action::ALL_IN}, 0, 2, 0);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.85f}, Action{1.025f}, Action{1.20f}, Action::ALL_IN}, 0, 2, 1, false);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.70f}, Action{0.875f}, Action{1.05f}, Action::ALL_IN}, 0, 2, 1, true);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.90f}, Action{1.075f}, Action{1.25f}, Action::ALL_IN}, 0, 2, 2, false);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.75f}, Action{0.935f}, Action{1.12f}, Action::ALL_IN}, 0, 2, 2, true);
  for(int pos = 3; pos < n_players; ++pos) {
    set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.815f}, Action{1.00f}, Action{1.185f}, Action::ALL_IN}, 0, 2, pos, false);
    set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.65}, Action{0.815f}, Action{1.00f}, Action::ALL_IN}, 0, 2, pos, true);
  }

  // preflop 4-bet
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.45f}, Action{0.55f}, Action{0.65f}, Action::ALL_IN}, 0, 3, 0);

  // flop
  set_actions({Action::CHECK_CALL, Action{0.50f}, Action{0.75f}, Action{1.00f}, Action::ALL_IN}, 1, 0, 0);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.50f}, Action{1.00f}, Action::ALL_IN}, 1, 1, 0);
  if(n_players <= 6) add_action(Action{0.75f}, 1, 1, 0);

  // turn
  set_actions({Action::CHECK_CALL, Action{0.50f}, Action{1.00f}, Action::ALL_IN}, 2, 0, 0);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{1.00f}, Action::ALL_IN}, 2, 1, 0);

  // river
  set_actions({Action::CHECK_CALL, Action{0.50f}, Action{1.00f}, Action::ALL_IN}, 3, 0, 0);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{1.00f}, Action::ALL_IN}, 3, 1, 0);
}

WPTGoldRingSimpleProfile::WPTGoldRingSimpleProfile(const int n_players) : ActionProfile{n_players} {
  // preflop RFI & isos
  for(int pos = 0; pos < 3; ++pos) {
    set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.85f}, Action::ALL_IN}, 0, 1, pos);
  }
  for(int pos = 3; pos < n_players; ++pos) {
    set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.52f}, Action::ALL_IN}, 0, 1, pos);
  }
  set_iso_actions({Action::FOLD, Action::CHECK_CALL, Action{0.70f}, Action::ALL_IN}, 0  );

  // preflop 3-bet
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{1.00f}, Action::ALL_IN}, 0, 2, 0);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{1.025f}, Action::ALL_IN}, 0, 2, 1, false);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.70f}, Action::ALL_IN}, 0, 2, 1, true);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{1.075f}, Action::ALL_IN}, 0, 2, 2, false);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.75f}, Action::ALL_IN}, 0, 2, 2, true);
  for(int pos = 3; pos < n_players; ++pos) {
    set_actions({Action::FOLD, Action::CHECK_CALL, Action{1.00f}, Action::ALL_IN}, 0, 2, pos, false);
    set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.65}, Action::ALL_IN}, 0, 2, pos, true);
  }

  // preflop 4-bet
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.55f}, Action::ALL_IN}, 0, 3, 0, false);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.45f}, Action::ALL_IN}, 0, 3, 0, true);

  // flop
  set_actions({Action::CHECK_CALL, Action{0.50f}, Action{0.75f}, Action{1.00f}, Action::ALL_IN}, 1, 0, 0);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.50f}, Action{1.00f}, Action::ALL_IN}, 1, 1, 0);

  // turn
  set_actions({Action::CHECK_CALL, Action{0.50f}, Action{1.00f}, Action::ALL_IN}, 2, 0, 0);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{1.00f}, Action::ALL_IN}, 2, 1, 0);

  // river
  set_actions({Action::CHECK_CALL, Action{1.00f}, Action::ALL_IN}, 3, 0, 0);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{1.00f}, Action::ALL_IN}, 3, 1, 0);
}

WPTGoldRingPreflopProfile::WPTGoldRingPreflopProfile(const int n_players) : ActionProfile{n_players} {
  // preflop RFI & isos
  for(int pos = 0; pos < 3; ++pos) {
    set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.85f}, Action::ALL_IN}, 0, 1, pos);
  }
  for(int pos = 3; pos < n_players; ++pos) {
    set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.52f}, Action::ALL_IN}, 0, 1, pos);
  }
  set_iso_actions({Action::FOLD, Action::CHECK_CALL, Action{0.70f}, Action::ALL_IN}, 0);

  // preflop 3-bet
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.83f}, Action{1.00f}, Action{1.17f}, Action::ALL_IN}, 0, 2, 0);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.85f}, Action{1.025f}, Action{1.20f}, Action::ALL_IN}, 0, 2, 1, false);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.70f}, Action{0.875f}, Action{1.05f}, Action::ALL_IN}, 0, 2, 1, true);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.90f}, Action{1.075f}, Action{1.25f}, Action::ALL_IN}, 0, 2, 2, false);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.75f}, Action{0.935f}, Action{1.12f}, Action::ALL_IN}, 0, 2, 2, true);
  for(int pos = 3; pos < n_players; ++pos) {
    set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.815f}, Action{1.00f}, Action{1.185f}, Action::ALL_IN}, 0, 2, pos, false);
    set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.65}, Action{0.815f}, Action{1.00f}, Action::ALL_IN}, 0, 2, pos, true);
  }

  // preflop 4-bet
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{0.45f}, Action{0.55f}, Action{0.65f}, Action::ALL_IN}, 0, 3, 0);

  // flop
  set_actions({Action::CHECK_CALL, Action{0.50f}, Action{1.00f}, Action::ALL_IN}, 1, 0, 0);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{1.00f}, Action::ALL_IN}, 1, 1, 0);

  // turn
  set_actions({Action::CHECK_CALL, Action{0.50f}, Action{1.00f}, Action::ALL_IN}, 2, 0, 0);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{1.00f}, Action::ALL_IN}, 2, 1, 0);

  // river
  set_actions({Action::CHECK_CALL, Action{1.00f}, Action::ALL_IN}, 3, 0, 0);
  set_actions({Action::FOLD, Action::CHECK_CALL, Action{1.00f}, Action::ALL_IN}, 3, 1, 0);
}

}
