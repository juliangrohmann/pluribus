#include <iostream>
#include <string>
#include <set>
#include <array>
#include <fstream>

#include <catch2/catch_test_macros.hpp>
#include <omp/Hand.h>
#include <omp/HandEvaluator.h>
#include <pluribus/poker.hpp>

using namespace pluribus;
using std::string;

TEST_CASE("Card indexing", "[card]") {
  int idx = 0;
  for(char rank : omp::RANKS) {
    for(char suit : omp::SUITS) {
      string card = string(1, rank) + suit;
      int card_idx = card_to_idx(card);
      REQUIRE(card_idx == idx++);
      REQUIRE(idx_to_card(card_idx) == card);
    }
  }
}
