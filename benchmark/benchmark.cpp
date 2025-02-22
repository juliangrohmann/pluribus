#ifdef UNIT_TEST

#include <iostream>
#include <string>
#include <set>
#include <array>
#include <fstream>
#include <cassert>

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <omp/Hand.h>
#include <omp/HandEvaluator.h>
#include <hand_isomorphism/hand_index.h>
#include <pluribus/poker.hpp>
#include <pluribus/cluster.hpp>
#include <pluribus/mccfr.hpp>

using namespace pluribus;
using std::string;

double calc_equity(const std::string& hero_str, omp::CardRange villain, const std::string& board_str) {
  omp::EquityCalculator eq;
  omp::CardRange hero_rng = omp::CardRange(hero_str);
  eq.start({hero_rng, villain}, omp::CardRange::getCardMask(board_str), 0, true);
  eq.wait();
  return eq.getResults().equity[0];
}

double simple_equity(const std::string& hero_str, const omp::CardRange villain, const std::string& board_str) {
  omp::Hand board = omp::Hand::empty() + omp::Hand(board_str);
  omp::Hand hero = omp::Hand(hero_str);
  return equity(hero, villain, board);
}

// TEST_CASE("Evaluate benchmark", "[eval]") {
//   omp::HandEvaluator evaluator;
//   omp::Hand hero = omp::Hand("Qd") + omp::Hand("As") + omp::Hand("6h") + omp::Hand("Js") + omp::Hand("2c");
//   omp::Hand villain = omp::Hand("3d") + omp::Hand("9h") + omp::Hand("Kc") + omp::Hand("4h") + omp::Hand("8s");
//   BENCHMARK("Eval 5 cards") {
//     bool winner = evaluator.evaluate(hero) > evaluator.evaluate(villain);
//   };
  
//   hero += omp::Hand("Jd");
//   villain += omp::Hand("4c");
//   BENCHMARK("Eval 6 cards") {
//     bool winner = evaluator.evaluate(hero) > evaluator.evaluate(villain);
//   };

//   hero += omp::Hand("6c");
//   villain += omp::Hand("Qc");
//   BENCHMARK("Eval 7 cards") {
//     bool winner = evaluator.evaluate(hero) > evaluator.evaluate(villain);
//   };
// };

TEST_CASE("Isomorphism unindex", "[iso]") {
  hand_indexer_t flop_indexer;
  uint8_t flop_cards[] = {2, 3};
  REQUIRE(hand_indexer_init(2, flop_cards, &flop_indexer));

  hand_indexer_t turn_indexer;
  uint8_t turn_cards[] = {2, 3, 1};
  REQUIRE(hand_indexer_init(3, turn_cards, &turn_indexer));

  hand_indexer_t river_indexer;
  uint8_t river_cards[] = {2, 3, 1, 1};
  REQUIRE(hand_indexer_init(4, river_cards, &river_indexer));

  uint8_t cards[7] = {};
  BENCHMARK("Flop") {
    hand_unindex(&flop_indexer, 1, 42, cards);
  };
  BENCHMARK("Turn") {
    hand_unindex(&turn_indexer, 2, 42, cards);
  };
  BENCHMARK("River") {
    hand_unindex(&river_indexer, 3, 42, cards);
  };

  hand_indexer_free(&flop_indexer);
  hand_indexer_free(&turn_indexer);
  hand_indexer_free(&river_indexer);
};

TEST_CASE("Equity calculation", "[equity]") {
  int category = 4;
  string hero = "9s4h";
  string flop = "3c5c2d";
  string turn = "3c5c2dQc";
  string river = "3c5c2dQcTs";
  BENCHMARK("Flop, simple") {
    simple_equity(hero, ochs_categories[category], flop);
  };
  BENCHMARK("Flop, calc") {
    calc_equity(hero, ochs_categories[category], flop);
  };
  BENCHMARK("Turn, simple") {
    simple_equity(hero, ochs_categories[category], turn);
  };
  BENCHMARK("Turn, calc") {
    calc_equity(hero, ochs_categories[category], turn);
  };
  BENCHMARK("River, simple") {
    simple_equity(hero, ochs_categories[category], river);
  };
  BENCHMARK("River, calc") {
    calc_equity(hero, ochs_categories[category], river);
  };
};

TEST_CASE("OCHS features", "[ochs]") {
  omp::EquityCalculator eq;
  string hero = "9s4h";
  string flop = "3c5c2d";
  string turn = "3c5c2dQc";
  string river = "3c5c2dQcTs";
  float feat[8];
  BENCHMARK("Flop") {
    assign_features(hero, flop, feat);
  };
  BENCHMARK("Turn") {
    assign_features(hero, turn, feat);
  };
  BENCHMARK("River") {
    assign_features(hero, river, feat);
  };
};

namespace pluribus {

void call_update_strategy(BlueprintTrainer& trainer, const PokerState& state, int i, const Board& board, 
                          const std::vector<Hand>& hands) {
  trainer.update_strategy(state, i, board, hands);
}

int call_traverse_mccfr(BlueprintTrainer& trainer, const PokerState& state, int i, const Board& board, 
                        const std::vector<Hand>& hands) {
  return trainer.traverse_mccfr(state, i, board, hands);
}

}

TEST_CASE("Blueprint trainer", "[mccfr]") {
  int n_players = 6;
  int n_chips= 10'000;
  int ante = 0;
  Board board{"AcTd2h3cQs"};
  std::vector<Hand> hands{Hand{"AsQs"}, Hand{"5c5h"}, Hand{"Kh5d"}, Hand{"Ah3d"}, Hand{"9s9h"}, Hand{"QhJd"}};
  BlueprintTrainer trainer{n_players, n_chips, ante};

  BENCHMARK("Update strategy") {
    call_update_strategy(trainer, PokerState{n_players, n_chips, ante}, 0, board, hands);
  };
  BENCHMARK("Traverse MCCFR") {
    call_traverse_mccfr(trainer, PokerState{n_players, n_chips, ante}, 0, board, hands);
  };
}

#endif