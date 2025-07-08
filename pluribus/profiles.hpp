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
};

class RingBlueprintProfile : public ActionProfile {
public:
  explicit RingBlueprintProfile(int n_players, int stack_size);
};

class WPTGoldRingBlueprintProfile : public ActionProfile {
public:
  explicit WPTGoldRingBlueprintProfile(int n_players);
};

class WPTGoldRingSimpleProfile : public ActionProfile {
public:
  explicit WPTGoldRingSimpleProfile(int n_players);
};

class WPTGoldRingPreflopProfile : public ActionProfile {
public:
  explicit WPTGoldRingPreflopProfile(int n_players);
};

}