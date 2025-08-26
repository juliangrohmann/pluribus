#ifdef UNIT_TEST

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <unistd.h>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/unordered_map.hpp>
#include <hand_isomorphism/hand_index.h>
#include <omp/Hand.h>
#include <omp/HandEvaluator.h>
#include <pluribus/actions.hpp>
#include <pluribus/agent.hpp>
#include <pluribus/blueprint.hpp>
#include <pluribus/cereal_ext.hpp>
#include <pluribus/cluster.hpp>
#include <pluribus/debug.hpp>
#include <pluribus/dist.hpp>
#include <pluribus/earth_movers_dist.hpp>
#include <pluribus/ev.hpp>
#include <pluribus/indexing.hpp>
#include <pluribus/mccfr.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/rng.hpp>
#include <pluribus/sampling.hpp>
#include <pluribus/simulate.hpp>
#include <pluribus/translate.hpp>
#include <pluribus/traverse.hpp>
#include <pluribus/util.hpp>

#include "lib.hpp"

using namespace pluribus;
using namespace testlib;
using Catch::Matchers::WithinAbs;
using std::string;
using std::cout;
using std::endl;

omp::Hand init_hand(const std::string& str) {
  return omp::Hand::empty() + omp::Hand(std::string(str));
}

std::vector<std::vector<int>> board_idx_sample(const int round, const int amount) {
  hand_indexer_t indexer;
  const int card_sum = init_indexer(indexer, round);
  const int n_idx = hand_indexer_size(&indexer, round);

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

std::vector<string> board_str_sample(const int round, const int amount) {
  std::vector<string> sample;
  sample.reserve(amount);
  for(const auto hands = board_idx_sample(round, amount); const auto& hand : hands) {
    string hand_str = "";
    for(const int idx : hand) {
      hand_str += idx_to_card(idx);
    }
    sample.push_back(hand_str);
  }
  return sample;
}

std::vector<omp::Hand> board_hand_sample(const int round, const int amount) {
  std::vector<omp::Hand> sample;
  sample.reserve(amount);
  for(const auto hands = board_idx_sample(round, amount); const auto& idx_hand : hands) {
    omp::Hand hand = omp::Hand::empty();
    for(const int idx : idx_hand) {
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
  const bool match = obj == loaded_obj;
  // unlink(fn.c_str());
  return match;
}

TEST_CASE("Card encode/decode", "[card]") {
  int idx = 0;
  for(const char rank : omp::RANKS) {
    for(const char suit : omp::SUITS) {
      string card = string(1, rank) + suit;
      const int card_idx = card_to_idx(card);
      REQUIRE(card_idx == idx++);
      REQUIRE(idx_to_card(card_idx) == card);
    }
  }
}

TEST_CASE("Hand contains", "[hand]") {
  for(const auto sample = board_idx_sample(1, 10'000); const auto& idx_hand : sample) {
    omp::Hand hand = omp::Hand::empty();
    for(const int idx : idx_hand) {
      hand += omp::Hand(idx);
    }
    std::vector<int> matches;
    for(int i = 0; i < MAX_CARDS; ++i) {
      const bool match_proper = hand.contains(omp::Hand::empty() + omp::Hand(i));
      const bool match_improper = hand.contains(omp::Hand(i));
      const bool should_match = std::find(idx_hand.data(), idx_hand.data() + idx_hand.size(), i) != idx_hand.data() + idx_hand.size();
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
  std::fstream file("../resources/eval_testset.txt");
  string line;
  while(std::getline(file, line)) {
    omp::HandEvaluator evaluator;
    std::erase(line, ' ');
    omp::Hand hero = omp::Hand::empty() + omp::Hand(line.substr(0, 10));
    omp::Hand villain = omp::Hand::empty() + omp::Hand(line.substr(10, 10));
    REQUIRE((evaluator.evaluate(hero) > evaluator.evaluate(villain)) == (line[line.length() - 1] == '1'));
  }
}

TEST_CASE("All-in sizing", "[actions]") {
  PokerState state{3, {2'000, 3'000, 1'000}, 0, false};
  state = state.apply({Action{0.60}, Action::CHECK_CALL, Action::CHECK_CALL, Action{0.50}});
  const int all_in_size = total_bet_size(state, Action::ALL_IN);
  REQUIRE(all_in_size == 1'750);
  REQUIRE(fractional_bet_size(state, total_bet_size(state, Action::ALL_IN)) == (all_in_size - 375.0) / 1500.0);
}

TEST_CASE("Pseudo harmonic action translation", "[actions]") {
  std::vector actions = {Action::CHECK_CALL, Action{0.33}, Action{0.50}, Action{0.75}, Action::ALL_IN};
  std::vector simple_actions = {Action{0.50}, Action{0.75}};
  PokerState state{3, {3'000, 3'000, 3'000}, 0, false};
  state = state.apply({Action{0.60}, Action::CHECK_CALL, Action::CHECK_CALL});
  REQUIRE(translate_pseudo_harmonic(Action{0.50}, actions, state) == Action{0.50});
  REQUIRE(translate_pseudo_harmonic(Action::ALL_IN, actions, state) == Action::ALL_IN);
  REQUIRE(translate_pseudo_harmonic(Action{1.00}, simple_actions, state) == Action{0.75});
  REQUIRE(translate_pseudo_harmonic(Action{0.25}, simple_actions, state) == Action{0.50});
  int N = 100'000;
  int n_A = 0;
  float A = 0.50f, B = 0.80f, x = 0.60f;
  for(int i = 0; i < N; ++i) {
    if(Action a = translate_pseudo_harmonic(Action{x}, {Action{A}, Action{B}}, state); a == Action{A}) ++n_A;
  }
  double p_A = (B - x) * (1 + A) / ((B - A) * (1 + x));
  REQUIRE(abs(static_cast<double>(n_A) / static_cast<double>(N) - p_A) < 0.01);
}

TEST_CASE("Simple equity solver", "[equity]") {
  for(const auto sample = board_str_sample(1, 10); const string& hand : sample) {
    string hero_str = hand.substr(0, 4);
    string board_str = hand.substr(4);
    auto hero = omp::Hand(hero_str);
    omp::Hand board = omp::Hand::empty() + omp::Hand(board_str);
    for(int i = 0; i < 8; ++i) {
      auto hero_rng = omp::CardRange(hero_str);
      auto villain = omp::CardRange(ochs_categories[i]);
      omp::EquityCalculator eq;
      eq.start({hero_rng, villain}, omp::CardRange::getCardMask(board_str), 0, true);
      eq.wait();
      const double calc_eq = eq.getResults().equity[0];
      const double simple_eq = equity(hero, villain, board);
      REQUIRE(abs(calc_eq - simple_eq) < 1e-5);
    }
  }
}

TEST_CASE("Simulate hands", "[poker][slow]") {
  constexpr int n_players = 9;
  constexpr int stack_size = 10'000;
  std::vector<RandomAgent> rng_agents;
  for(int i = 0; i < n_players; ++i) rng_agents.push_back(RandomAgent{RingBlueprintProfile{n_players}});
  std::vector<Agent*> agents;
  for(int i = 0; i < n_players; ++i) agents.push_back(&rng_agents[i]);
  const auto results = simulate(agents, PokerConfig{n_players, 0, false}, stack_size, 100'000);
  long net = 0l;
  for(const long result : results) {
    net += result;
  }
  REQUIRE(net == 0l);
}

TEST_CASE("Straddle", "[poker]") {
  PokerState state{6, 20'000, 50, true};
  REQUIRE(state.get_pot().total() == 650);
  REQUIRE(state.get_active() == 3);

  auto history = ActionHistory{{Action::FOLD, Action::FOLD, Action::CHECK_CALL, Action::FOLD, Action::CHECK_CALL}};
  state = state.apply(history);
  REQUIRE(state.get_round() == 0);
  REQUIRE(state.get_active() == 2);
}

TEST_CASE("Split pot", "[poker]") {
  Deck deck;
  const std::vector hands{Hand{"KsTc"}, Hand{"As4c"}, Hand{"Ac2h"}};
  const Board board{"AdKh9s9h5c"};
  const ActionHistory actions = {
    Action{0.8f}, Action::FOLD, Action::CHECK_CALL,
    Action::CHECK_CALL, Action{0.33f}, Action{1.00f}, Action::CHECK_CALL,
    Action::CHECK_CALL, Action::CHECK_CALL,
    Action::CHECK_CALL, Action::CHECK_CALL
  };
  const auto result = simulate_round(board, hands, actions, PokerConfig{static_cast<int>(hands.size()), 0, false}, 10'000);
  REQUIRE(result[0] == -50);
  REQUIRE(result[1] == 25);
  REQUIRE(result[2] == 25);
}

std::vector<int> utility_vector(const SlimPokerState& state, const Board& board, const std::vector<Hand>& hands, const std::vector<int>& chips,
    const RakeStructure& rake) {
  const omp::HandEvaluator eval;
  std::vector<int> util;
  for(int i = 0; i < 3; ++i) util.push_back(utility(state, i, board, hands, chips[i], rake, eval));
  return util;
}

TEST_CASE("Side pot", "[poker]") {
  const std::vector hands{Hand{"QcQh"}, Hand{"KcKh"}, Hand{"AcAh"}};
  const Board board{"2c2h2d2s3h"};
  const std::vector chips = {2'000, 1'000, 500};
  const SlimPokerState state{3, chips, 0, false};
  auto cover_allin = state.apply_copy({
    Action::CHECK_CALL, Action::CHECK_CALL, Action::CHECK_CALL,
    Action::ALL_IN, Action::CHECK_CALL, Action::CHECK_CALL
  });
  SlimPokerState leftover_allin = state.apply_copy({
    Action::CHECK_CALL, Action::CHECK_CALL, Action::CHECK_CALL,
    Action::CHECK_CALL, Action::CHECK_CALL, Action::ALL_IN, Action::CHECK_CALL, Action::ALL_IN, Action::CHECK_CALL
  });
  const RakeStructure no_rake{0.0, 0};
  const auto expected_util = std::vector{-1'000, 0, 1'000};
  REQUIRE(utility_vector(cover_allin, board, hands, chips, no_rake) == expected_util);
  REQUIRE(utility_vector(leftover_allin, board, hands, chips, no_rake) == expected_util);
}

std::vector<int> pokerkit_utiltiies(const std::string& line, const int n_players) {
  std::istringstream iss(line);
  std::vector<int> util(n_players);
  for(int i = 0; i < n_players; ++i) {
    if(!(iss >> util[i])) Logger::error("Error: expected " + std::to_string(n_players) +" integers on line: " + line);
  }
  return util;
}

TEST_CASE("Utility test set", "[poker]") {
  const std::filesystem::path dir = std::filesystem::path{".."} / "resources";
  const std::vector<std::string> fns = {"utility_no_sidepots, utility_sidepots"};
  for(const std::string& fn : fns) {
    std::ifstream pokerkit_file((dir / fn).string() + ".pokerkit");
    if(!pokerkit_file.is_open()) Logger::error("Failed to opoen pokerkit file.");
    UtilityTestSet test_set;
    cereal_load(test_set, (dir / fn).string() + ".testset");
    const omp::HandEvaluator eval;
    std::string line;
    int it = 0;
    for(const auto& test_case : test_set.cases) {
      std::getline(pokerkit_file, line);
      auto pokerkit_util = pokerkit_utiltiies(line, test_case.state.get_players().size());
      SlimPokerState terminal = test_case.state.apply_copy(test_case.actions);;
      std::vector<int> util(test_case.state.get_players().size());
      for(int i = 0; i < test_case.state.get_players().size(); ++i) {
        const Player& p = test_case.state.get_players()[i];
        // TODO: mismatch due to different odd chip assignment in split side pots. need to collapse side pots at showdown, see side_pot_payoff
        REQUIRE(abs(utility(terminal, i, test_case.board, test_case.hands, p.get_chips() + p.get_betsize(), test_set.rake, eval) - pokerkit_util[i]) <= 2);
      }
      ++it;
    }
  }
}

TEST_CASE("VPIP", "[poker]") {
  const PokerState state{6, 10'000, 0};
  REQUIRE(state.vpip_players() == 0);
  REQUIRE(state.has_player_vpip(0) == false);
  REQUIRE(state.has_player_vpip(1) == false);
  REQUIRE(state.has_player_vpip(2) == false);
  REQUIRE(state.is_in_position(state.get_active()) == true);

  PokerState limp_state = state.apply({Action::CHECK_CALL, Action::FOLD, Action::FOLD, Action::FOLD, Action::CHECK_CALL});
  REQUIRE(limp_state.vpip_players() == 2);
  REQUIRE(limp_state.is_in_position(limp_state.get_active()) == false);

  PokerState ip_state = state.apply({Action{1.00f}, Action::FOLD, Action::FOLD});
  REQUIRE(ip_state.vpip_players() == 1);
  REQUIRE(ip_state.has_player_vpip(2) == true);
  REQUIRE(ip_state.is_in_position(ip_state.get_active()) == true);
}

TEST_CASE("Action profile", "[profile]") {
  ActionProfile profile;
  const std::vector iso = {Action::FOLD, Action{2.00f}, Action::ALL_IN};
  const std::vector sb = {Action::FOLD, Action{0.50f}, Action{0.80f}, Action::ALL_IN};
  const std::vector lowjack = {Action::FOLD, Action{0.22f}, Action{0.33f}, Action::ALL_IN};
  const std::vector cutoff_oop = {Action::FOLD, Action{0.60f}, Action::ALL_IN};
  const std::vector cutoff_ip = {Action::FOLD, Action{0.25f}, Action{1.20f}, Action{1.40f}, Action::ALL_IN};
  const std::vector bb_oop = {Action::FOLD, Action{0.65f}, Action{0.90f}, Action::ALL_IN};
  const std::vector bb_ip = {Action::FOLD, Action{0.30f}, Action{0.75f}, Action::ALL_IN};
  profile.set_iso_actions(iso, 0, false);
  profile.set_iso_actions(iso, 0, true);
  profile.set_actions(sb, 0, 0, 0);
  profile.set_actions(bb_oop, 0, 0, 1);
  profile.set_actions(bb_ip, 0, 0, 1, true);
  profile.set_actions(lowjack, 0, 0, 2);
  profile.set_actions(cutoff_oop, 0, 0, 4);
  profile.set_actions(cutoff_ip, 0, 0, 4, true);

  const PokerState state{6, 10'000, 0}; // pos == 2
  REQUIRE(profile.get_actions(state) == lowjack);

  const PokerState limp_state = state.apply(Action::CHECK_CALL);
  REQUIRE(profile.get_actions(limp_state) == iso);

  const PokerState ip_overflow_state = state.apply({Action::FOLD, Action::FOLD, Action{1.00f}}); // pos == 5
  REQUIRE(profile.get_actions(ip_overflow_state) == cutoff_ip); // pos overflow

  const PokerState bb_oop_state = ip_overflow_state.apply({Action::FOLD, Action::FOLD}); // pos = 1
  REQUIRE(profile.get_actions(bb_oop_state) == bb_oop);

  const PokerState bb_ip_state = state.apply({Action::FOLD, Action::FOLD, Action::FOLD, Action::FOLD, Action{1.00f}}); // pos = 1
  REQUIRE(profile.get_actions(bb_ip_state) == bb_ip);
}

void test_hand_distribution(const PokerRange& range, const std::function<Hand()>& sampler, const long n_samples, const double threshold) {
  const double n_combos = range.n_combos();
  std::unordered_map<Hand, long> sampled;
  for(long i = 0; i < n_samples; ++i) {
    sampled[sampler()] += 1;
  }
  for(auto& [hand, amount] : sampled) {
    std::cout << "n_combos=" << n_combos << ", n_samples=" << n_samples << ", sampled=" << amount << "\n";
    const double rel_freq = amount * n_combos / n_samples;
    std::cout << std::fixed << std::setprecision(2) << hand.to_string() << ": " << rel_freq << " (" << range.frequency(hand) << ")\n";
    REQUIRE(abs(rel_freq - range.frequency(hand)) < threshold);
  }
}

void test_biased_freq(const std::vector<Action>& actions, const std::vector<float>& freq, const Action bias, const float factor, std::vector<int> biased_idxs) {
  const auto b_freq = biased_freq(actions, freq, bias, factor);
  float b_sum = 0.0f, norm_sum = 0.0f;
  std::vector<float> correct_freq;
  for(int fidx = 0; fidx < b_freq.size(); ++fidx) {
    correct_freq.push_back(std::ranges::find(biased_idxs, fidx) != biased_idxs.end() ? freq[fidx] * factor : freq[fidx]);
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
  const std::vector no_fold = {Action::CHECK_CALL, Action{0.50f}, Action::ALL_IN};
  const std::vector no_bet = {Action::FOLD, Action::CHECK_CALL};
  const std::vector facing_bet = {Action::FOLD, Action::CHECK_CALL, Action{0.30f}, Action{0.80f}, Action::ALL_IN};
  const std::vector facing_check = {Action::CHECK_CALL, Action{0.30f}, Action{0.50f}, Action{1.50f}, Action::ALL_IN};
  const std::vector freq_2 = {0.25f, 0.75f};
  const std::vector freq_3 = {0.10f, 0.25f, 0.65f};
  const std::vector freq_5 = {0.10f, 0.25f, 0.15f, 0.30f, 0.20f};
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
    single_clusters[round] = BlueprintClusterMap::get_instance()->cluster(round, board, hand);
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
      const uint16_t prog_cluster = BlueprintClusterMap::get_instance()->cluster(round, prog_indices[round]);
      REQUIRE(prog_cluster == single_clusters[round]);
    }
  }
}

TEST_CASE("Flop indexing", "[index]") {
  std::unordered_set<hand_index_t> indexes;
  hand_index_t min_idx = std::numeric_limits<hand_index_t>::max();
  hand_index_t max_idx = std::numeric_limits<hand_index_t>::min();
  for(uint8_t c1 = 0; c1 < MAX_CARDS; ++c1) {
    for(uint8_t c2 = 0; c2 < MAX_CARDS; ++c2) {
      for(uint8_t c3 = 0; c3 < MAX_CARDS; ++c3) {
        if(c1 == c2 || c1 == c3 || c2 == c3) continue;
        const uint8_t cards[3] = {c1, c2, c3};
        hand_index_t idx = FlopIndexer::get_instance()->index(cards);
        indexes.insert(idx);
        min_idx = std::min(idx, min_idx);
        max_idx = std::max(idx, max_idx);
      }
    }
  }
  REQUIRE(indexes.size() == NUM_DISTINCT_FLOPS);
  REQUIRE(min_idx == 0);
  REQUIRE(max_idx == NUM_DISTINCT_FLOPS - 1);
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
      const uint16_t cached_cluster = BlueprintClusterMap::get_instance()->cluster(round, indexer.index(board, hand, round));
      REQUIRE(cached_cluster == single_clusters[round]);
    }
  }
}

void test_sampler_mask(RoundSampler& sampler, const SamplingMode mode, const std::vector<uint8_t>& dead_cards) {
  sampler.set_mode(mode);
  const auto sample = sampler.sample();
  auto mask = card_mask(dead_cards);
  for(const auto& hand : sample.hands) mask |= hand.mask();
  REQUIRE(sample.mask == mask);
}

TEST_CASE("Round sampler", "[sampling][slow]") {
  constexpr int n_samples = 10'000'000;
  const auto dead_cards = str_to_cards("AcTh3d2s");
  std::vector<PokerRange> ranges;
  for(int i = 0; i < 2; ++i) ranges.push_back(PokerRange::random());
  RoundSampler sampler{ranges, dead_cards};
  auto sample_fun = [&sampler](auto& dist) {  
    auto sample = sampler.sample();
    dist.add_hand(sample.hands[0], sample.weight);
  };
  sampler.set_mode(SamplingMode::MARGINAL_REJECTION);
  const auto marginal_rejection_1 = build_distribution(n_samples, sample_fun, false);
  const auto marginal_rejection_2 = build_distribution(n_samples, sample_fun, false);
  sampler.set_mode(SamplingMode::IMPORTANCE_REJECTION);
  const auto importance_rejection = build_distribution(n_samples, sample_fun, false);
  sampler.set_mode(SamplingMode::IMPORTANCE_RANDOM_WALK);
  const auto importance_walk = build_distribution(n_samples, sample_fun, false);
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
  PokerState state{bp.get_config().poker, 10'000};
  std::vector actions = {Action{0.75f}, Action::CHECK_CALL, Action::CHECK_CALL, Action{0.50f}, Action::CHECK_CALL, Action::CHECK_CALL, Action::CHECK_CALL};
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
  const ActionHistory actions = {
    Action{0.8f}, Action::FOLD, Action::CHECK_CALL,
    Action::CHECK_CALL, Action{0.33f}, Action{1.00f}, Action::CHECK_CALL,
  };
  PokerState state{3};
  state = state.apply(actions);

  REQUIRE(test_serialization(state));
}

TEST_CASE("Serialize ActionHistory", "[serialize]") {
  const ActionHistory actions = {
    Action{0.8f}, Action::FOLD, Action::CHECK_CALL,
    Action::CHECK_CALL, Action{0.33f}, Action{1.00f}, Action::CHECK_CALL,
    Action::CHECK_CALL, Action::CHECK_CALL,
    Action::CHECK_CALL, Action::CHECK_CALL
  };
  REQUIRE(test_serialization(actions));
}

TEST_CASE("Serialize TreeBlueprintSolver", "[serialize][blueprint][slow]") {
  TreeBlueprintSolver trainer{SolverConfig{PokerConfig{2, 0, false}, HeadsUpBlueprintProfile{10'000}}};
  trainer.solve(1'000'000);
  
  REQUIRE(test_serialization(trainer));
}

TEST_CASE("EMD heuristic - partial mass", "[emd]") {
    constexpr int C = 2;
    const std::vector x = {0, 0}; // both points in cluster 0
    const std::vector x_w = {1.00, 0.00};
    const std::vector m_w = {0.75, 0.25};

    const std::vector<std::vector<std::pair<double, int>>> sorted_distances = {{
        {{0.0, 0}, {1.0, 1}}, // sorted ascending
        {{0.0, 1}, {1.0, 0}}
    }};

    double cost = emd_heuristic(x, x_w, m_w, sorted_distances);
    // First point gets all from mean 0 (0 cost)
    // Second point gets 0.25 from mean 0 (0 cost) + 0.25 from mean 1 (cost 0.25)
    REQUIRE_THAT(cost, WithinAbs(0.25, 1e-12));
}

TEST_CASE("EMD heuristic - zero distances", "[emd]") {
    constexpr int C = 3;
    const std::vector x = {0, 1, 2};
    const std::vector w = {1.0/3, 1.0/3, 1.0/3};

    std::vector<std::vector<std::pair<double, int>>> sorted_distances;
    for (int pc = 0; pc < C; ++pc) {
        sorted_distances[pc] = {{0.0, pc}, {0.0, (pc+1)%C}, {0.0, (pc+2)%C}}; // all zero distances
    }

    double cost = emd_heuristic(x, w, w, sorted_distances);
    REQUIRE_THAT(cost, WithinAbs(0.0, 1e-12));
}

TEST_CASE("EMD heuristic - mismatched sizes logs error", "[emd]") {
    constexpr int C = 2;
    const std::vector x = {0, 1};
    const std::vector w = {0.5, 0.5};

    // Wrong length in sorted_distances[0]
    const std::vector<std::vector<std::pair<double, int>>> sorted_distances = {{
        {{0.0, 0}}, // invalid
        {{0.0, 1}, {1.0, 0}}
    }};
    REQUIRE_THROWS(emd_heuristic(x, w, w, sorted_distances));
}

TEST_CASE("EMD heuristic - closest cluster is not self", "[emd]") {
  constexpr int C = 3;
  const std::vector x = {0, 1, 2};
  const std::vector w = {1.0/3, 1.0/3, 1.0/3};

  // Distances are sorted ascending per point cluster
  const std::vector<std::vector<std::pair<double, int>>> sorted_distances = {{
    {{0.5, 2}, {1.0, 0}, {2.0, 1}}, // from cluster 0: closest mean cluster is #2 (0.5 away)
    {{0.2, 0}, {0.7, 2}, {1.5, 1}}, // from cluster 1: closest mean cluster is #0 (0.2 away)
    {{0.1, 1}, {0.3, 0}, {1.0, 2}}  // from cluster 2: closest mean cluster is #1 (0.1 away)
  }};
  double cost = emd_heuristic(x, w, w, sorted_distances);
  REQUIRE_THAT(cost, WithinAbs(1.0/6.0 + 1.0/15.0 + 1.0/30.0, 1e-12));
}


#endif