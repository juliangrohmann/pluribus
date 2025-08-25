#include <chrono>
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

void build_state_testset(const int n_players, const ActionProfile& profile, const std::string& fn,
    const long n_iter = 100'000, const int min_chips = 2'000, const int max_chips = 20'000, const bool sidepots = true) {
  Logger::log("Initializing HandIndexer...");
  Logger::log(HandIndexer::get_instance() ? "Success." : "Failure.");
  std::vector<RandomAgent> rng_agents;
  for(int p = 0; p < n_players; ++p) rng_agents.push_back(RandomAgent{profile});
  std::vector<Agent*> agents;
  for(int p = 0; p < n_players; ++p) agents.push_back(&rng_agents[p]);
  ShowdownTestSet testset{profile};
  testset.cases.resize(n_iter);
  const auto t_0 = std::chrono::high_resolution_clock::now();
  const long log_interval = n_iter / 100L;
  #pragma omp parallel for schedule(dynamic, 1)
  for(long it = 0; it < n_iter; ++it) {
    thread_local omp::HandEvaluator eval;
    thread_local RakeStructure no_rake{0.0, 0};
    thread_local Board board;
    thread_local MarginalRejectionSampler sampler{std::vector(n_players, PokerRange::full())};
    if(it > 0 && it % log_interval == 0) Logger::log(progress_str(it, n_iter, t_0));
    const RoundSample sample = sampler.sample();
    board = sample_board({}, sample.mask);
    std::vector<int> chips = get_random_chips(n_players, min_chips, max_chips, sidepots);
    SlimPokerState state(n_players, chips, 0, false);
    ShowdownTestCase& curr_case = testset.cases[it];
    while(!state.is_terminal()) {
      Action a = agents[state.get_active()]->act(state, board, sample.hands[state.get_active()]);
      curr_case.actions.push_back(a);
      state.apply_in_place(a);
    }
    for(int p = 0; p < n_players; ++p) curr_case.utilities.push_back(utility(state, p, board, sample.hands, chips[p], no_rake, eval));
  }
  cereal_save(testset, fn);
}

int main(const int argc, char* argv[]) {
  if(argc < 2) std::cout << "Usage: ./TestSet path/to/out/fn.bin [n_players, n_iterations, min_chips, max_chips, --sidepots]";
  else {
    const int n_players = argc > 2 ? atoi(argv[2]) : 6;
    const int n_iterations = argc > 3 ? atoi(argv[3]) : 100'000;
    const int min_chips = argc > 4 ? atoi(argv[4]) : 2'000;
    const int max_chips = argc > 5 ? atoi(argv[5]) : 20'000;
    const bool sidepots = argc > 6 ? strcmp(argv[6], "--sidepots") == 0 : false;
    if(n_players <= 2) throw std::runtime_error("n_players >= 3 required");
    build_state_testset(n_players, RingBlueprintProfile{n_players}, argv[1], n_iterations, min_chips, max_chips, sidepots);
  }
}