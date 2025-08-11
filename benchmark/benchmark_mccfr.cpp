#include <iostream>
#include <chrono>
#include <omp/HandEvaluator.h>
#include <pluribus/sampling.hpp>
#include <pluribus/mccfr.hpp>

using namespace pluribus;

namespace pluribus {

template <template<typename> class StorageT>
int call_traverse_mccfr(MCCFRSolver<StorageT>* trainer, const PokerState& state, int i, const Board& board, 
    const std::vector<Hand>& hands, std::vector<CachedIndexer>& indexers, const omp::HandEvaluator& eval, std::ostringstream& debug) {
  return trainer->traverse_mccfr(state, 1, i, board, hands, indexers, eval, trainer->init_regret_storage(), debug);
}

}

int main(int argc, char* argv[]) {
  long n = argc > 1 ? atoi(argv[1]) : 1'000'000'000L;

  SolverConfig config{PokerConfig{2, 0, false}, HeadsUpBlueprintProfile{10'000}};
  omp::HandEvaluator eval;
  TreeBlueprintSolver trainer{config, BlueprintSolverConfig{}};
  std::ostringstream debug;
  // TODO: allocate all storage nodes

  // preload cache
  HandIndexer::get_instance();
  BlueprintClusterMap::get_instance();

  RoundSampler sampler{config.init_ranges, config.init_board};
  RoundSample sample = sampler.sample();
  auto t_0 = std::chrono::high_resolution_clock::now();
  int step = 5000;
  for(long i = 0; i < n; ++i) {
    if(i > 0 && i % step == 0) {
      auto t_i = std::chrono::high_resolution_clock::now();
      long dt_us = std::chrono::duration_cast<std::chrono::microseconds>(t_i - t_0).count();
      double us_per_it = static_cast<double>(dt_us) / i;
      std::cout << std::fixed << std::setprecision(1) << std::setw(6) << i / 1'000.0 << "k it " 
                << std::setw(6) << dt_us / 1000 << " ms, " 
                << std::setprecision(2) << std::setw(8) << us_per_it << " us/it, " 
                << std::setprecision(1) << std::setw(10) << 1'000'000.0 / us_per_it << " it/sec\n";
    }
    sampler.next_sample(sample);
    Board board = sample_board(config.init_board, sample.mask);
    std::vector<CachedIndexer> indexers(config.poker.n_players);
    for(int h_idx = 0; h_idx < sample.hands.size(); ++h_idx) {
      indexers[h_idx].index(board, sample.hands[h_idx], 3);
    }
    call_traverse_mccfr(&trainer, config.init_state, 0, board, sample.hands, indexers, eval, debug);
  }
}
