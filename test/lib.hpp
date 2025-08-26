#pragma once

#include <vector>
#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>
#include <pluribus/poker.hpp>

using namespace pluribus;

namespace testlib {
  struct UtilityTestCase {
    SlimPokerState state;
    std::vector<Hand> hands;
    Board board;
    ActionHistory actions;
    std::vector<int> utilities; // TODO: unused, remove and re-generate test set

    template <class Archive>
    void serialize(Archive& ar) {
      ar(state, hands, board, utilities, actions);
    }
  };

  struct UtilityTestSet {
    ActionProfile profile;
    RakeStructure rake;
    std::vector<UtilityTestCase> cases;

    template <class Archive>
    void serialize(Archive& ar) {
      ar(profile, rake, cases);
    }
  };
}

