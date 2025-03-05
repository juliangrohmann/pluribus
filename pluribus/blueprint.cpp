#include <sys/sysinfo.h>
#include <pluribus/cereal_ext.hpp>
#include <pluribus/mccfr.hpp>
#include <pluribus/blueprint.hpp>

namespace pluribus {

void distill(const std::string& preflop_fn, const std::vector<std::string>& postflop_fns) {
  // _freq()
  for(int i = 0; i < postflop_fns.size(); ++i) {
    const auto& bp = cereal_load<BlueprintTrainer>(postflop_fns[i]);
  }
  

}

}
