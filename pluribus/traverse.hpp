#pragma once

#include <pluribus/range_viewer.hpp>
#include <pluribus/mccfr.hpp>
#include <string>

namespace pluribus {

void traverse(RangeViewer* viewer, const std::string& bp_fn);

std::unordered_map<Action, RenderableRange> trainer_ranges(const BlueprintTrainer& bp, const PokerState& state, 
  const Board& board, PokerRange& base_range, bool force_regrets);

}

