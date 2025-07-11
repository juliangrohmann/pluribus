#ifdef UNIT_TEST

#include <iostream>
#include <string>
#include <set>
#include <array>
#include <fstream>
#include <cassert>

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <omp/Random.h>
#include <omp/Hand.h>
#include <omp/HandEvaluator.h>
#include <omp/EquityCalculator.h>
#include <hand_isomorphism/hand_index.h>
#include <pluribus/poker.hpp>
#include <pluribus/rng.hpp>
#include <pluribus/cluster.hpp>
#include <pluribus/sampling.hpp>
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

template <template<typename> class StorageT>
void call_update_strategy(BlueprintSolver<StorageT>* trainer, const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands,
    StorageT<int>* regret_storage, StorageT<float>* avg_storage, std::ostringstream& debug) {
  trainer->update_strategy(state, i, board, hands, trainer->init_regret_storage(), trainer->init_avg_storage(), debug);
}

template <template<typename> class StorageT>
int call_traverse_mccfr(MCCFRSolver<StorageT>* trainer, const PokerState& state, int i, const Board& board, 
    const std::vector<Hand>& hands, std::vector<CachedIndexer>& indexers, const omp::HandEvaluator& eval, std::ostringstream& debug) {
  return trainer->traverse_mccfr(state, 1, i, board, hands, eval, trainer->init_regret_storage(), trainer->init_avg_storage(), debug);
}
// int call_traverse_mccfr(MCCFRTrainer* trainer, const PokerState& state, int i, const Board& board, 
//                         const std::vector<Hand>& hands, const omp::HandEvaluator& eval, std::ostringstream& debug) {
//   return trainer->traverse_mccfr(state, 1, i, board, hands, eval, debug);
// }

}

// TEST_CASE("Blueprint trainer", "[mccfr]") {
//   PokerConfig config{6, 10'000, 0};
//   omp::HandEvaluator eval;
//   Board board{"AcTd2h3cQs"};
//   std::vector<Hand> hands{Hand{"AsQs"}, Hand{"5c5h"}, Hand{"Kh5d"}, Hand{"Ah3d"}, Hand{"9s9h"}, Hand{"QhJd"}};
//   // std::vector<Hand> hands{Hand{"AsQs"}, Hand{"5c5h"}};
//   MappedBlueprintSolver trainer{BlueprintSolverConfig{}, SolverConfig{config}};
//   trainer.set_log_level(SolverLogLevel::NONE);
//   // trainer.allocate_all();
//   std::ostringstream debug;

//   BENCHMARK("Update strategy") {
//     call_update_strategy(&trainer, PokerState{config}, 0, board, hands, debug);
//   };
//   BENCHMARK("Traverse MCCFR") {
//     call_traverse_mccfr(&trainer, PokerState{config}, 0, board, hands, eval, debug);
//   };
// }

TEST_CASE("GSL discrete sampling", "[sampling]") {
  auto sparse_range = PokerRange();
  sparse_range.add_hand(Hand{"AcAh"}, 0.5);
  sparse_range.add_hand(Hand{"AcKh"}, 1.0);
  sparse_range.add_hand(Hand{"2c2h"}, 0.25);
  GSLDiscreteDist dist{sparse_range.weights()};

  double sum = 0.0;
  BENCHMARK("Sample") {
    HoleCardIndexer::get_instance()->hand(dist.sample());
  };
}

TEST_CASE("Round sampling", "[sampling]") {
  std::vector<int> n_players = {3, 6, 9};
  std::vector<uint8_t> dead_cards = {42, 10, 33, 22};
  std::vector<std::vector<PokerRange>> ranges(n_players.size());
  std::vector<RoundSampler> samplers;
  for(int ni = 0; ni < n_players.size(); ++ni) {
    for(int i = 0; i < n_players[ni]; ++i) ranges[ni].push_back(PokerRange::random());
    samplers.push_back(RoundSampler{ranges[ni], dead_cards});
  }
  
  for(int ni = 0; ni < n_players.size(); ++ni) {
    std::string title = std::to_string(n_players[ni]) + " players, " + std::to_string(dead_cards.size()) + " dead cards";
    samplers[ni].set_mode(SamplingMode::MARGINAL_REJECTION);
    BENCHMARK(title + ", rejection sampling") {
      samplers[ni].sample();
    };
    samplers[ni].set_mode(SamplingMode::IMPORTANCE_REJECTION);
    BENCHMARK(title + ", importance rejection sampling") {
      samplers[ni].sample();
    };
    samplers[ni].set_mode(SamplingMode::IMPORTANCE_RANDOM_WALK);
    BENCHMARK(title + ", importance random-walk sampling") {
      samplers[ni].sample();
    };

    auto sample = samplers[ni].sample();
    BENCHMARK(title + ", importance random-walk sampling (in-place)") {
      samplers[ni].next_sample(sample);
    };
  }
}

TEST_CASE("Fast uniform int sampling (OMP)", "[sampling]") {
  omp::XoroShiro128Plus rng{std::random_device{}()};
  omp::FastUniformIntDistribution<unsigned,21> dist(0, MAX_COMBOS - 1);
  BENCHMARK("Sample") {
    dist(rng);
  };
}

#endif