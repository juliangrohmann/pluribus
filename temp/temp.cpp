#include <iostream>
#include <fstream>
#include <array>
#include <cmath>
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
  BlueprintTrainer trainer{9, 10'000, 0};
  std::cout << trainer.count_infosets() << "\n";
  trainer.mccfr_p(10'000'000'000);
  trainer.save_strategy("6p_100bb_t10B.bin");
  return 0;
}