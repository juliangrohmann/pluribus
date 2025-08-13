#pragma once

#include <pluribus/actions.hpp>

namespace pluribus {

class BiasActionProfile : public ActionProfile {
public:
  BiasActionProfile();
};

class HeadsUpBlueprintProfile : public ActionProfile {
public:
  explicit HeadsUpBlueprintProfile(int stack_size);
  static constexpr int max_actions() { return 8; }
};

class RingBlueprintProfile : public ActionProfile {
public:
  explicit RingBlueprintProfile(int n_players);
  static constexpr int max_actions() { return 11; }
};

class WPTGoldRingBlueprintProfile : public ActionProfile {
public:
  explicit WPTGoldRingBlueprintProfile(int n_players, bool fine_grained);
  static constexpr int max_actions() { return 10; }
};

}