#include <iostream>
#include <string>
#include <set>
#include <array>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>

#include <catch2/catch_test_macros.hpp>
#include <omp/Hand.h>
#include <omp/HandEvaluator.h>
#include <hand_isomorphism/hand_index.h>
#include <pluribus/poker.hpp>
#include <pluribus/cluster.hpp>

using namespace pluribus;
using std::string;
using std::cout;
using std::endl;

omp::Hand init_hand(const std::string& str) {
  return omp::Hand::empty() + omp::Hand(std::string(str));
}

std::vector<std::vector<int>> board_idx_sample(int round, int amount) {
  hand_indexer_t indexer;
  uint8_t n_cards[round + 1];
  uint8_t all_rounds[] = {2, 3, 1, 1};
  int card_sum = 0;
  for(int i = 0; i < round + 1; ++i) {
    n_cards[i] = all_rounds[i];
    card_sum += all_rounds[i];
  }

  assert(hand_indexer_init(round + 1, n_cards, &indexer));
  int n_idx = hand_indexer_size(&indexer, round);

  uint8_t cards[7] = {};
  std::vector<std::vector<int>> hands;
  for(int idx = 0; idx < n_idx; idx += n_idx / amount) {
    hand_unindex(&indexer, round, idx, cards);
    std::vector<int> hand;
    hand.reserve(card_sum);
    for(int i = 0; i < card_sum; ++i) {
      hand.push_back(cards[i]);
    }
    hands.push_back(hand);
  }
  hand_indexer_free(&indexer);
  return hands;
}

std::vector<string> board_str_sample(int round, int amount) {
  std::vector<string> sample;
  sample.reserve(amount);
  auto hands = board_idx_sample(round, amount);
  for(const auto& hand : hands) {
    string hand_str = "";
    for(int idx : hand) {
      hand_str += idx_to_card(idx);
    }
    sample.push_back(hand_str);
  }
  return sample;
}

std::vector<omp::Hand> board_hand_sample(int round, int amount) {
  std::vector<omp::Hand> sample;
  sample.reserve(amount);
  auto hands = board_idx_sample(round, amount);
  for(const auto& idx_hand : hands) {
    omp::Hand hand = omp::Hand::empty();
    for(int idx : idx_hand) {
      hand += omp::Hand(idx);
    }
    sample.push_back(hand);
  }
  return sample;
}

TEST_CASE("Card encode/decode", "[card]") {
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

TEST_CASE("Hand contains", "[hand]") {
  auto sample = board_idx_sample(1, 10'000);
  for(const auto& idx_hand : sample) {
    omp::Hand hand = omp::Hand::empty();
    for(int idx : idx_hand) {
      hand += omp::Hand(idx);
    }
    std::vector<int> matches;
    for(int i = 0; i < 52; ++i) {
      bool match_proper = hand.contains(omp::Hand::empty() + omp::Hand(i));
      bool match_improper = hand.contains(omp::Hand(i));
      bool should_match = (std::find(idx_hand.data(), idx_hand.data() + idx_hand.size(), i) != idx_hand.data() + idx_hand.size());
      REQUIRE(match_proper == should_match);
      REQUIRE(match_improper == should_match);
      if(match_proper) {
        matches.push_back(i);
      }
    }
    REQUIRE(matches.size() == idx_hand.size());
  }
}

TEST_CASE("Hand intersection", "[hand]") {
  REQUIRE(init_hand("AcQcTs").contains(init_hand("Qc8s")));
  REQUIRE(init_hand("AcQcTs").contains(init_hand("TsTc")));
  REQUIRE(!init_hand("3hQcTs").contains(init_hand("3s8s")));
  REQUIRE(!init_hand("JhJcJd").contains(init_hand("Js8s")));
}

TEST_CASE("Evaluate hand", "[eval]") {
  omp::HandEvaluator evaluator;
  std::fstream file("../resources/eval_testset.txt");
  string line;
  while(std::getline(file, line)) {
    line.erase(std::remove(line.begin(), line.end(), ' '), line.end());
    omp::Hand hero = omp::Hand::empty() + omp::Hand(line.substr(0, 10));
    omp::Hand villain = omp::Hand::empty() + omp::Hand(line.substr(10, 10));
    REQUIRE((evaluator.evaluate(hero) > evaluator.evaluate(villain)) == (line[line.length() - 1] == '1'));
  }
};

TEST_CASE("Simple equity solver", "[equity]") {
  auto sample = board_str_sample(1, 10);
  for(const string& hand : sample) {
    string hero_str = hand.substr(0, 4);
    string board_str = hand.substr(4);
    omp::Hand hero = omp::Hand(hero_str);
    omp::Hand board = omp::Hand::empty() + omp::Hand(board_str);
    for(int i = 0; i < 8; ++i) {
      omp::CardRange hero_rng = omp::CardRange(hero_str);
      omp::CardRange villain = omp::CardRange(ochs_categories[i]);
      omp::EquityCalculator eq;
      eq.start({hero_rng, villain}, omp::CardRange::getCardMask(board_str), 0, true);
      eq.wait();
      double calc_eq = eq.getResults().equity[0];
      double simple_eq = equity(hero, villain, board);
      REQUIRE(abs(calc_eq - simple_eq) < 1e-5);
    }
  }
}
