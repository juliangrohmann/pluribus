#include <iostream>
#include <string>
#include <set>
#include <array>
#include <fstream>

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <omp/Hand.h>
#include <omp/HandEvaluator.h>
#include <pluribus/poker.hpp>

using std::string;

TEST_CASE("Decode card", "[card]") {
  std::set<int> card_ids;
  for(int rank = 0; rank < 13; ++rank) {
    for(int suit = 0; suit < 4; ++suit) {
      string card_str = string(1, "23456789TJQKA"[rank]) + "sdhc"[suit];
      int bits = make_card(card_str);
      card_ids.insert(bits);
      REQUIRE(decode_cards(bits) == card_str);
    }
  }
  REQUIRE(card_ids.size() == 52);
};

TEST_CASE("Decode hand", "[card]") {
  string ranks = "23456789TJQKA";
  string suits = "sdhc";
  for(int i = 0; i < 13; ++i) {
    for(int j = 0; j < 13; ++j) {
      int si = i % 4;
      int sj = j % 4;
      string hand_str = string(1, ranks[i]) + suits[si] + ranks[j] + suits[sj];
      REQUIRE(decode_cards(make_hand(hand_str)) == hand_str);
    }
  }
}

TEST_CASE("Evaluate hand", "[eval]") {
  omp::HandEvaluator evaluator;
  std::fstream file("../resources/eval_testset.txt");
  string line;
  while(std::getline(file, line)) {
    std::string hero_str = line.substr(0, 15);
    std::string villain_str = line.substr(15, 15);
    omp::Hand hero = omp::Hand::empty();
    omp::Hand villain = omp::Hand::empty();
    for(int i = 0; i < hero_str.length(); i += 3) {
      hero += omp::Hand(hero_str.substr(i, 2));
    }
    for(int i = 0; i < villain_str.length(); i += 3) {
      villain += omp::Hand(villain_str.substr(i, 2));
    }
    REQUIRE((evaluator.evaluate(hero) > evaluator.evaluate(villain)) == (line[line.length() - 1] == '1'));
  }
};

TEST_CASE("Evaluate benchmark", "[eval]") {
  omp::HandEvaluator evaluator;
  omp::Hand hero = omp::Hand("Qd") + omp::Hand("As") + omp::Hand("6h") + omp::Hand("Js") + omp::Hand("2c");
  omp::Hand villain = omp::Hand("3d") + omp::Hand("9h") + omp::Hand("Kc") + omp::Hand("4h") + omp::Hand("8s");
  BENCHMARK("Eval 5 cards") {
    bool winner = evaluator.evaluate(hero) > evaluator.evaluate(villain);
  };
  
  hero += omp::Hand("Jd");
  villain += omp::Hand("4c");
  BENCHMARK("Eval 6 cards") {
    bool winner = evaluator.evaluate(hero) > evaluator.evaluate(villain);
  };

  hero += omp::Hand("6c");
  villain += omp::Hand("Qc");
  BENCHMARK("Eval 7 cards") {
    bool winner = evaluator.evaluate(hero) > evaluator.evaluate(villain);
  };
};