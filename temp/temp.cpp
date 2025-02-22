#include <iostream>
#include <fstream>
#include <array>
#include <cmath>
#include <chrono>
#include <unordered_map>
#include <tqdm/tqdm.hpp>
#include <hand_isomorphism/hand_index.h>
#include <omp/Hand.h>
#include <omp/CardRange.h>
#include <omp/EquityCalculator.h>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>
#include <cnpy.h>
#include <pluribus/cluster.hpp>
#include <pluribus/util.hpp>
#include <pluribus/debug.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/actions.hpp>
#include <pluribus/infoset.hpp>
#include <pluribus/simulate.hpp>
#include <pluribus/mccfr.hpp>

using namespace std;
using namespace pluribus;

int main(int argc, char* argv[]) {
  BlueprintTrainer trainer{6, 10'000, 0};
  auto t0 = std::chrono::high_resolution_clock::now();
  trainer.mccfr_p(1'000'000);
  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout << "dt=" << std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count() << "\n";
  trainer.save_strategy("6p_100bb_t10B.bin");
  return 0;
}
