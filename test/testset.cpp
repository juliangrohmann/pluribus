#include <chrono>
#include <fstream>
#include <vector>
#include <pluribus/agent.hpp>
#include <pluribus/logging.hpp>
#include <pluribus/mccfr.hpp>
#include <pluribus/rng.hpp>
#include <pluribus/sampling.hpp>

#include "lib.hpp"
#include "pluribus/mccfr.hpp"
#include "pluribus/profiles.hpp"

using namespace pluribus;
using namespace testlib;

std::vector<int> get_random_chips(const int n_players, const int min_stack, const int max_stack, const bool sidepots) {
  std::uniform_int_distribution stack_dist{min_stack, max_stack};
  std::vector<int> chips;
  if(sidepots) {
    for(int i = 0; i < n_players; ++i) chips.push_back(stack_dist(GlobalRNG::instance()));
  }
  else {
    chips.resize(n_players, stack_dist(GlobalRNG::instance()));
  }
  return chips;
}

std::string to_token(const SlimPokerState& state, const Action a) {
  if(a == Action::FOLD) return "F";
  if(a == Action::CHECK_CALL) return "C";
  return std::to_string(total_bet_size(state, a));
}

void build_state_testset(const int n_players, const ActionProfile& profile, const std::string& fn,
    const long n_iter = 100'000, const int min_chips = 2'000, const int max_chips = 20'000, const bool sidepots = true) {
  Logger::log("Initializing HoleCardIndexer...");
  Logger::log(HoleCardIndexer::get_instance() ? "Success." : "Failure.");
  std::vector<RandomAgent> rng_agents;
  for(int p = 0; p < n_players; ++p) rng_agents.push_back(RandomAgent{profile});
  std::vector<Agent*> agents;
  for(int p = 0; p < n_players; ++p) agents.push_back(&rng_agents[p]);
  std::ofstream pokerkit_file(std::filesystem::path(fn).parent_path() / "testset.pokerkit");
  if(!pokerkit_file) Logger::log("Failed to open pokerkit output file.");
  UtilityTestSet test_set{profile, RakeStructure{0.0, 0}};
  test_set.cases.resize(n_iter);
  MarginalRejectionSampler sampler{std::vector(n_players, PokerRange::full())};
  const auto t_0 = std::chrono::high_resolution_clock::now();
  const long log_interval = n_iter / 100L;
  for(long it = 0; it < n_iter; ++it) {
    if(it > 0 && it % log_interval == 0) Logger::log(progress_str(it, n_iter, t_0));
    UtilityTestCase& curr_case = test_set.cases[it];
    const RoundSample sample = sampler.sample();
    curr_case.hands = sample.hands;
    curr_case.board = sample_board({}, sample.mask);
    std::vector<int> chips = get_random_chips(n_players, min_chips, max_chips, sidepots);
    pokerkit_file << it << " ";
    for(int stack : chips) pokerkit_file << stack << " ";
    for(const Hand& hand : curr_case.hands) pokerkit_file << hand.to_string() << " ";
    pokerkit_file << curr_case.board.to_string() << " D0 ";
    SlimPokerState state(n_players, chips, 0, false);
    curr_case.state = state;
    int pokerkit_round = 0;
    while(!state.is_terminal()) {
      Action a = agents[state.get_active()]->act(state, curr_case.board, sample.hands[state.get_active()]);
      curr_case.actions.push_back(a);
      pokerkit_file << to_token(state, a) << " ";
      state.apply_in_place(a);
      while(state.get_round() > pokerkit_round && pokerkit_round < 3) {
        pokerkit_file << "D" << ++pokerkit_round << " ";
      }
    }
    pokerkit_file << "\n";
  }
  cereal_save(test_set, fn);
}

int main(const int argc, char* argv[]) {
  if(argc == 1) std::cout << "Usage: ./TestSet [n_players, n_iterations, min_chips, max_chips, --sidepots path/to/out/fn]";
  else {
    const int n_players = argc > 1 ? atoi(argv[1]) : 6;
    const int n_iterations = argc > 2 ? atoi(argv[2]) : 100'000;
    const int min_chips = argc > 3 ? atoi(argv[3]) : 2'000;
    const int max_chips = argc > 4 ? atoi(argv[4]) : 20'000;
    const bool sidepots = argc > 5 ? strcmp(argv[5], "--sidepots") == 0 : false;
    const std::string fn = argc > 6 ? argv[6] : std::string{"../resources/utility_"} + (sidepots ? "" : "no_") + "sidepots.testset";
    if(n_players <= 2) throw std::runtime_error("n_players >= 3 required");
    build_state_testset(n_players, RingBlueprintProfile{n_players}, fn, n_iterations, min_chips, max_chips, sidepots);
  }
}