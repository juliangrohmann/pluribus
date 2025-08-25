#pragma once

#include <vector>
#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>
#include <pluribus/poker.hpp>

using namespace pluribus;

namespace testlib {
  struct ShowdownTestCase {
    SlimPokerState state;
    ActionHistory actions;
    std::vector<int> utilities;

    template <class Archive>
    void serialize(Archive& ar) {
      ar(state, actions, utilities);
    }
  };

  struct ShowdownTestSet {
    ActionProfile profile;
    std::vector<ShowdownTestCase> cases;

    template <class Archive>
    void serialize(Archive& ar) {
      ar(cases, profile);
    }
  };
}

