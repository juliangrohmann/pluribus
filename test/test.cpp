#ifdef UNIT_TEST

#include <iostream>
#include <string>
#include <set>
#include <array>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <unistd.h>
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
#include <pluribus/storage.hpp>
#include <pluribus/blueprint.hpp>
#include <pluribus/traverse.hpp>
#include <pluribus/sampling.hpp>
#include <pluribus/dist.hpp>
#include <pluribus/ev.hpp>
#include <pluribus/mccfr.hpp>
#include <pluribus/cereal_ext.hpp>
#include <pluribus/util.hpp>
#include <pluribus/debug.hpp>
#include <pluribus/rng.hpp>

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

template <class T>
bool test_serialization(const T& obj) {
  std::string fn = "test_serialization.bin";
  cereal_save(obj, fn);
  T loaded_obj;
  cereal_load(loaded_obj, fn);
  bool match = (obj == loaded_obj);
  // unlink(fn.c_str());
  return match;
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
    for(int i = 0; i < MAX_CARDS; ++i) {
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

TEST_CASE("Simulate hands", "[poker][slow]") {
  int n_players = 9;
  int stack_size = 10'000;
  std::vector<RandomAgent> rng_agents;
  for(int i = 0; i < n_players; ++i) rng_agents.push_back(RandomAgent{BlueprintActionProfile{n_players, stack_size}});
  std::vector<Agent*> agents;
  for(int i = 0; i < n_players; ++i) agents.push_back(&rng_agents[i]);
  auto results = simulate(agents, PokerConfig{n_players, stack_size, 0}, 100'000);
  long net = 0l;
  for(long result : results) {
    net += result;
  }
  REQUIRE(net == 0l);
}

TEST_CASE("Straddle", "[poker]") {
  PokerState state{6, 20'000, 50, true};
  REQUIRE(state.get_pot() == 650);
  REQUIRE(state.get_active() == 3);

  auto history = ActionHistory{{Action::FOLD, Action::FOLD, Action::CHECK_CALL, Action::FOLD, Action::CHECK_CALL}};
  state = state.apply(history);
  REQUIRE(state.get_round() == 0);
  REQUIRE(state.get_active() == 3);
}
TEST_CASE("Split pot", "[poker]") {
  Deck deck;
  std::vector<Hand> hands{Hand{"KsTc"}, Hand{"As4c"}, Hand{"Ac2h"}};
  Board board{"AdKh9s9h5c"};
  ActionHistory actions = {
    Action{0.8f}, Action::FOLD, Action::CHECK_CALL,
    Action::CHECK_CALL, Action{0.33f}, Action{1.00f}, Action::CHECK_CALL,
    Action::CHECK_CALL, Action::CHECK_CALL,
    Action::CHECK_CALL, Action::CHECK_CALL
  };
  auto result = simulate_round(board, hands, actions, PokerConfig{static_cast<int>(hands.size()), 10'000, 0});
  REQUIRE(result[0] == -50);
  REQUIRE(result[1] == 25);
  REQUIRE(result[2] == 25);
}

void test_hand_distribution(const PokerRange& range, std::function<Hand()>sampler, long n_samples, double threshold) {
  double n_combos = range.n_combos();
  std::unordered_map<Hand, long> sampled;
  for(long i = 0; i < n_samples; ++i) {
    sampled[sampler()] += 1;
  }
  for(auto& entry : sampled) {
    std::cout << "n_combos=" << n_combos << ", n_samples=" << n_samples << ", sampled=" << entry.second << "\n";
    double rel_freq = static_cast<double>(entry.second * n_combos) / n_samples;
    std::cout << std::fixed << std::setprecision(2) << entry.first.to_string() << ": " << rel_freq << " (" << range.frequency(entry.first) << ")\n";
    REQUIRE(abs(rel_freq - range.frequency(entry.first)) < threshold);
  }
}

// TEST_CASE("Sample PokerRange with dead_cards", "[range]") {
//   std::uniform_real_distribution<float> dist(0.0f, 1.0f);
//   PokerRange range;
//   uint8_t dead_card = card_to_idx("Ad");
//   range.add_hand(Hand{dead_card, card_to_idx("Jd")});
//   range.add_hand(Hand{card_to_idx("9h"), dead_card});
//   range.add_hand(Hand{"AdAh"});
//   range.add_hand(Hand{"AcAd"});
//   range.add_hand(Hand{"AcTh"});
//   range.add_hand(Hand{"8s8h"});

//   std::unordered_map<Hand, float> sampled;
//   for(int i = 0; i < 10'000; ++i) {
//     Hand hand = range.sample({dead_card});
//     REQUIRE(hand.cards()[0] != dead_card);
//     REQUIRE(hand.cards()[1] != dead_card);
//   }
// }

// void test_hand_sampler(const PokerRange& range, bool sparse, long n, double thresh) {
//   HandSampler sampler{range, sparse};
//   test_hand_distribution(range, [&]() -> Hand { return sampler.sample(); }, n, thresh);
// }

// TEST_CASE("Sample hands with HandSampler", "[sampler]") {
//   auto sparse_range = PokerRange();
//   sparse_range.add_hand({"AcAh"}, 0.5);
//   sparse_range.add_hand({"AcKh"}, 1.0);
//   sparse_range.add_hand({"2c2h"}, 0.25);
//   test_hand_sampler(sparse_range, false, 1'000'000, 0.01);
//   test_hand_sampler(sparse_range, true, 1'000'000, 0.01);
//   test_hand_sampler(PokerRange::random(), false, 10'000'000, 0.04);
// }

void test_biased_freq(std::vector<Action> actions, std::vector<float> freq, Action bias, float factor, std::vector<int> biased_idxs) {
  auto b_freq = biased_freq(actions, freq, bias, factor);
  float b_sum = 0.0f, norm_sum = 0.0f;
  std::vector<float> correct_freq;
  for(int fidx = 0; fidx < b_freq.size(); ++fidx) {
    correct_freq.push_back(std::find(biased_idxs.begin(), biased_idxs.end(), fidx) != biased_idxs.end() ?
                                     freq[fidx] * factor : freq[fidx]);
    b_sum += b_freq[fidx];
    norm_sum += correct_freq[fidx];
  }
  for(int fidx = 0; fidx < correct_freq.size(); ++fidx) {
    correct_freq[fidx] /= norm_sum;
    REQUIRE(b_freq[fidx] == correct_freq[fidx]);
  }
  REQUIRE(abs(b_sum - 1.0f) < 0.001);
}

TEST_CASE("Bias action frequencies", "[bias]") {
  std::vector<Action> no_fold = {Action::CHECK_CALL, Action{0.50f}, Action::ALL_IN};
  std::vector<Action> no_bet = {Action::FOLD, Action::CHECK_CALL};
  std::vector<Action> facing_bet = {Action::FOLD, Action::CHECK_CALL, Action{0.30f}, Action{0.80f}, Action::ALL_IN};
  std::vector<Action> facing_check = {Action::CHECK_CALL, Action{0.30f}, Action{0.50f}, Action{1.50f}, Action::ALL_IN};
  float factor = 0.5f;
  std::vector<float> freq_2 = {0.25f, 0.75f};
  std::vector<float> freq_3 = {0.10f, 0.25f, 0.65f};
  std::vector<float> freq_5 = {0.10f, 0.25f, 0.15f, 0.30f, 0.20f};
  test_biased_freq(no_fold, freq_3, Action::BIAS_FOLD, 5.0f, {});
  test_biased_freq(no_fold, freq_3, Action::BIAS_CALL, 5.0f, {0});
  test_biased_freq(no_fold, freq_3, Action::BIAS_RAISE, 5.0f, {1, 2});
  test_biased_freq(no_bet, freq_2, Action::BIAS_RAISE, 5.0f, {});
  test_biased_freq(no_bet, freq_2, Action::BIAS_CALL, 5.0f, {1});
  test_biased_freq(facing_bet, freq_5, Action::BIAS_RAISE, 5.0f, {2, 3, 4});
  test_biased_freq(facing_check, freq_5, Action::BIAS_RAISE, 5.0f, {1, 2, 3, 4});
}

std::array<uint16_t, 4> independent_indices(const Board& board, const Hand& hand) {
  std::array<uint16_t, 4> single_clusters;
  for(int round = 0; round < 4; ++round) {
    single_clusters[round] = FlatClusterMap::get_instance()->cluster(round, board, hand);
  }
  return single_clusters;
}

TEST_CASE("Progressive indexing", "[index]") {
  Deck deck;
  for(int i = 0; i < 100; ++i) {
    deck.reset();
    Board board{deck};
    Hand hand{deck};
    auto single_clusters = independent_indices(board, hand);

    auto cards = collect_cards(board, hand);
    hand_index_t prog_indices[4];
    hand_index_all(HandIndexer::get_instance()->get_indexer(3), cards.data(), prog_indices);
    for(int round = 0; round < 4; ++round) {
      uint16_t prog_cluster = FlatClusterMap::get_instance()->cluster(round, prog_indices[round]);
      REQUIRE(prog_cluster == single_clusters[round]);
    }
  }
}

TEST_CASE("Cached indexing", "[index]") {
  Deck deck;
  for(int i = 0; i < 100; ++i) {
    deck.reset();
    Board board{deck};
    Hand hand{deck};
    auto single_clusters = independent_indices(board, hand);

    auto cards = collect_cards(board, hand);
    auto indexer = CachedIndexer(3);
    for(int round = 0; round < 4; ++round) {
      uint16_t cached_cluster = FlatClusterMap::get_instance()->cluster(round, indexer.index(board, hand, round));
      REQUIRE(cached_cluster == single_clusters[round]);
    }
  }
}

void test_sampler_mask(RoundSampler& sampler, SamplingMode mode, const std::vector<uint8_t> dead_cards) {
  sampler.set_mode(mode);
  auto sample = sampler.sample();
  auto mask = card_mask(dead_cards);
  for(const auto& hand : sample.hands) mask |= hand.mask();
  REQUIRE(sample.mask == mask);
}

TEST_CASE("Round sampler", "[sampling][slow]") {
  int n_samples = 10'000'000;
  auto dead_cards = str_to_cards("AcTh3d2s");
  std::vector<PokerRange> ranges;
  for(int i = 0; i < 2; ++i) ranges.push_back(PokerRange::random());
  RoundSampler sampler{ranges, dead_cards};
  auto sample_fun = [&sampler](auto& dist) {  
    auto sample = sampler.sample();
    dist[sample.hands[0]] += sample.weight;
  };
  sampler.set_mode(SamplingMode::MARGINAL_REJECTION);
  auto marginal_rejection_1 = build_distribution(n_samples, sample_fun, false);
  auto marginal_rejection_2 = build_distribution(n_samples, sample_fun, false);
  sampler.set_mode(SamplingMode::IMPORTANCE_REJECTION);
  auto importance_rejection = build_distribution(n_samples, sample_fun, false);
  sampler.set_mode(SamplingMode::IMPORTANCE_RANDOM_WALK);
  auto importance_walk = build_distribution(n_samples, sample_fun, false);
  REQUIRE(distribution_rmse(marginal_rejection_1, marginal_rejection_2) < 0.0006);
  REQUIRE(distribution_rmse(marginal_rejection_1, importance_rejection) < 0.00075);
  REQUIRE(distribution_rmse(marginal_rejection_1, importance_walk) < 0.00075);
  test_sampler_mask(sampler, SamplingMode::MARGINAL_REJECTION, dead_cards);
  test_sampler_mask(sampler, SamplingMode::IMPORTANCE_REJECTION, dead_cards);
  test_sampler_mask(sampler, SamplingMode::IMPORTANCE_RANDOM_WALK, dead_cards);
}

TEST_CASE("Lossless monte carlo EV", "[ev][slow][dependency]") {
  long N = 10'000'000;
  LosslessBlueprint bp;
  cereal_load(bp, "lossless_bp_2p_100bb_0ante");
  std::vector<uint8_t> board = str_to_cards("AcTd3c2s");
  PokerState state{bp.get_config().poker};
  std::vector<Action> actions = {Action{0.75f}, Action::CHECK_CALL, Action::CHECK_CALL, Action{0.50f}, Action::CHECK_CALL, Action::CHECK_CALL, Action::CHECK_CALL};
  state = state.apply(ActionHistory{actions});
  auto ranges = build_ranges(state.get_action_history().get_history(), Board{board}, bp);
  MonteCarloEV ev_solver{};
  double enum_ev = enumerate_ev(bp, state, 0, ranges, board);
  ResultEV mc_result = ev_solver.set_max_iterations(N)
      ->set_min_iterations(N)
      ->lossless(&bp, state, 0, ranges, board);
  REQUIRE(abs(enum_ev - mc_result.ev) / enum_ev < 0.03);
}

TEST_CASE("Serialize Hand", "[serialize]") {
  REQUIRE(test_serialization(Hand{"Ac2s"}));
  REQUIRE(test_serialization(Hand{"3h5h"}));
  REQUIRE(test_serialization(Hand{"3c3s"}));
  REQUIRE(test_serialization(Hand{4, 1}));
  REQUIRE(test_serialization(Hand{50, 22}));
  REQUIRE(test_serialization(Hand{21, 32}));
}

TEST_CASE("Serialize PokerState", "[serialize]") {
  ActionHistory actions = {
    Action{0.8f}, Action::FOLD, Action::CHECK_CALL,
    Action::CHECK_CALL, Action{0.33f}, Action{1.00f}, Action::CHECK_CALL,
  };
  PokerState state{3};
  state = state.apply(actions);

  REQUIRE(test_serialization(state));
}

TEST_CASE("Serialize ActionHistory", "[serialize]") {
  ActionHistory actions = {
    Action{0.8f}, Action::FOLD, Action::CHECK_CALL,
    Action::CHECK_CALL, Action{0.33f}, Action{1.00f}, Action::CHECK_CALL,
    Action::CHECK_CALL, Action::CHECK_CALL,
    Action::CHECK_CALL, Action::CHECK_CALL
  };
  REQUIRE(test_serialization(actions));
}

TEST_CASE("Serialize StrategyStorage, BlueprintSolver", "[serialize][blueprint][slow]") {
  MappedBlueprintSolver trainer{};
  trainer.solve(100'000);

  REQUIRE(test_serialization(trainer.get_strategy()));
  REQUIRE(test_serialization(trainer));
}

TEST_CASE("Serialize TreeBlueprintSolver", "[serialize][blueprint][slow][inconsistent]") {
  TreeBlueprintSolver trainer{};
  trainer.solve(1'000'000);
  
  REQUIRE(test_serialization(trainer));
}

#endif