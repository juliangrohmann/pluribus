#ifdef UNIT_TEST

#include <iostream>
#include <string>
#include <set>
#include <array>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cereal/archives/binary.hpp>
#include <cereal/types/unordered_map.hpp>
#include <catch2/catch_test_macros.hpp>
#include <omp/Hand.h>
#include <omp/HandEvaluator.h>
#include <hand_isomorphism/hand_index.h>
#include <pluribus/poker.hpp>
#include <pluribus/cluster.hpp>
#include <pluribus/agent.hpp>
#include <pluribus/simulate.hpp>
#include <pluribus/actions.hpp>
#include <pluribus/infoset.hpp>
#include <pluribus/mccfr.hpp>
#include <pluribus/util.hpp>

using namespace pluribus;
using std::string;
using std::cout;
using std::endl;

omp::Hand init_hand(const std::string& str) {
  return omp::Hand::empty() + omp::Hand(std::string(str));
}

std::vector<std::vector<int>> board_idx_sample(int round, int amount) {
  hand_indexer_t indexer;
  int card_sum = init_indexer(indexer, round);
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

TEST_CASE("Simulate hands", "[poker]") {
  std::vector<Agent*> agents;
  for(int i = 0; i < 9; ++i) {
    agents.push_back(new RandomAgent());
  }
  auto results = simulate(agents, 10'000, 0, 100'000);
  long net = 0l;
  for(long result : results) {
    net += result;
  }
  REQUIRE(net == 0l);
}

TEST_CASE("Split pot", "[poker]") {
  Deck deck;
  std::vector<Hand> hands{{"KsTc"}, {"As4c"}, {"Ac2h"}};
  Board board{"AdKh9s9h5c"};
  ActionHistory actions = {
    Action::PREFLOP_2_BET, Action::FOLD, Action::CHECK_CALL,
    Action::CHECK_CALL, Action::BET_33, Action::POSTFLOP_2_BET, Action::CHECK_CALL,
    Action::CHECK_CALL, Action::CHECK_CALL,
    Action::CHECK_CALL, Action::CHECK_CALL
  };
  auto result = simulate_round(board, hands, actions, 10000, 0);
  REQUIRE(result[0] == -50);
  REQUIRE(result[1] == 25);
  REQUIRE(result[2] == 25);
}

TEST_CASE("Serialize ActionHistory", "[serialize]") {
  ActionHistory actions = {
    Action::PREFLOP_2_BET, Action::FOLD, Action::CHECK_CALL,
    Action::CHECK_CALL, Action::BET_33, Action::POSTFLOP_2_BET, Action::CHECK_CALL,
    Action::CHECK_CALL, Action::CHECK_CALL,
    Action::CHECK_CALL, Action::CHECK_CALL
  };
  
  {
    std::ofstream os("test_actions.bin", std::ios::binary);
    cereal::BinaryOutputArchive oarchive(os);
    oarchive(actions);
  }
  
  ActionHistory loaded_actions;
  {
    std::ifstream is("test_actions.bin", std::ios::binary);
    cereal::BinaryInputArchive iarchive(is);
    iarchive(loaded_actions);
  }

  REQUIRE(loaded_actions.size() == actions.size());
  for(int i = 0; i < loaded_actions.size(); ++i) {
    REQUIRE(loaded_actions.get(i) == actions.get(i));
  }
}

TEST_CASE("Serialize InformationSet", "[serialize]") {
  Hand hand{"KsTc"};
  Board board{"AdKh9s9h5c"};
  ActionHistory actions = {
    Action::PREFLOP_2_BET, Action::FOLD, Action::CHECK_CALL,
    Action::CHECK_CALL, Action::BET_33, Action::POSTFLOP_2_BET, Action::CHECK_CALL,
    Action::CHECK_CALL
  };
  InformationSet info_set{actions, board, hand, 2, 3, 10'000, 0};

  {
    std::ofstream os("test_info_set.bin", std::ios::binary);
    cereal::BinaryOutputArchive oarchive(os);
    oarchive(info_set);
  }
  
  InformationSet loaded_info_set;
  {
    std::ifstream is("test_info_set.bin", std::ios::binary);
    cereal::BinaryInputArchive iarchive(is);
    iarchive(loaded_info_set);
  }

  REQUIRE(loaded_info_set == info_set);
}

TEST_CASE("Serialize RegretStorage", "[serialize]") {
  BlueprintTrainer trainer{2, 10'000, 0};
  trainer.mccfr_p(1000);

  {
    std::ofstream os("test_strategy.bin", std::ios::binary);
    cereal::BinaryOutputArchive oarchive(os);
    oarchive(trainer.get_regrets());
  }
  
  int n_histories = HistoryIndexer::size(trainer.get_n_players(), trainer.get_n_chips(), trainer.get_ante());
  RegretStorage loaded_regrets{trainer.get_n_players(), trainer.get_n_chips(), trainer.get_ante(), 200, n_histories};
  {
    std::ifstream is("test_strategy.bin", std::ios::binary);
    cereal::BinaryInputArchive iarchive(is);
    iarchive(loaded_regrets);
  }

  REQUIRE(loaded_regrets == trainer.get_regrets());
}

#endif