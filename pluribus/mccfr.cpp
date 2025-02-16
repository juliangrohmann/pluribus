#include <iostream>
#include <iomanip>
#include <pluribus/debug.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/mccfr.hpp>

namespace pluribus {

BlueprintTrainer::BlueprintTrainer(int strategy_interval, int prune_thresh, int lcfr_thresh, int discount_interval) :
    _strategy_interval{strategy_interval}, _prune_thresh{prune_thresh}, _lcfr_thresh{lcfr_thresh}, _discount_interval{discount_interval},
    _strategy{} {

  std::cout << "Strategy interval: " << _strategy_interval << "\n";
  std::cout << "Prune threshold: " << _prune_thresh << "\n";
  std::cout << "LCFR threshold: " << _lcfr_thresh << "\n";
  std::cout << "Discount interval: " << _discount_interval << "\n";
}

void BlueprintTrainer::mccfr_p(int T) {
  for(long t = 0; t < T; ++t) {
    if(t % _strategy_interval == 0) {
      if(verbose) std::cout << "Updating strategy...\n";
    }
  }
}

}

