#pragma once

#include <iostream>
#include <iomanip>
#include <unordered_map>
#include <vector>
#include <string>
#include <pluribus/range.hpp>
#include <pluribus/actions.hpp>
#include <pluribus/mccfr.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/cluster.hpp>
#include <pluribus/range_viewer.hpp>
#include <pluribus/decision.hpp>
#include <pluribus/util.hpp>
#include <pluribus/blueprint.hpp>

namespace pluribus {

void traverse_trainer(RangeViewer* viewer_p, const std::string& bp_fn);
void traverse_tree(RangeViewer* viewer_p, const std::string& bp_fn);
void traverse_blueprint(RangeViewer* viewer_p, const std::string& bp_fn);
Action str_to_action(const std::string& str);
void render_ranges(RangeViewer* viewer_p, const PokerRange& base_range, const std::unordered_map<Action, RenderableRange>& action_ranges);
PokerRange build_action_range(const PokerRange& base_range, const Action& a, const PokerState& state, const Board& board,
    const DecisionAlgorithm& decision);
void update_ranges(std::vector<PokerRange>& ranges, Action a, const PokerState& state, const Board& board, 
    const DecisionAlgorithm& decision); 
std::vector<PokerRange> build_ranges(const std::vector<Action>& actions, const Board& board, const Strategy<float>& strat);

std::unordered_map<Action, RenderableRange> build_renderable_ranges(const DecisionAlgorithm& decision, const ActionProfile& profile, 
    const PokerState& state, const Board& board, PokerRange& base_range);
}

