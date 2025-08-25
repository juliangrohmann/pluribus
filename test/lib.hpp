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
    std::vector<int> utilities;

    template <class Archive>
    void serialize(Archive& ar) {
      // std::cout << "State:\n";
      ar(state);
      // std::cout << "Hands:\n";
      ar(hands);
      // std::cout << "Board:\n";
      ar(board);
      // std::cout << "Utilities:\n";
      ar(utilities);
      // std::cout << "Actions:\n";
      ar(actions);
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

