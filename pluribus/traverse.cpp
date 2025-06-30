#include <iostream>
#include <iomanip>
#include <unordered_map>
#include <vector>
#include <functional>
#include <pluribus/range.hpp>
#include <pluribus/mccfr.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/cluster.hpp>
#include <pluribus/range_viewer.hpp>
#include <pluribus/util.hpp>
#include <pluribus/traverse.hpp>

namespace pluribus {

void traverse_trainer(RangeViewer* viewer_p, const std::string& bp_fn) {
  std::cout << "Loading trainer from " << bp_fn << " for traversal... " << std::flush;
  MappedBlueprintSolver bp;
  cereal_load(bp, bp_fn);
  std::cout << "Success.\n";
  traverse(viewer_p, bp);
}

void traverse_blueprint(RangeViewer* viewer_p, const std::string& bp_fn) {
  std::cout << "Loading blueprint from " << bp_fn << " for traversal... " << std::flush;
  LosslessBlueprint bp;
  cereal_load(bp, bp_fn);
  std::cout << "Success.\n";
  traverse(viewer_p, bp);
}

Action str_to_action(const std::string& str) {
  if(str.starts_with("check") || str.starts_with("call")) return Action::CHECK_CALL;
  else if(str.starts_with("fold")) return Action::FOLD;
  else if(str.starts_with("all-in")) return Action::ALL_IN;
  else if(str.starts_with("bet")) return Action(std::stoi(str.substr(4)) / 100.0);
  else throw std::runtime_error("Invalid action string: " + str);
}

void render_ranges(RangeViewer* viewer_p, const PokerRange& base_range, const std::unordered_map<Action, RenderableRange>& action_ranges) {
  std::vector<RenderableRange> ranges{RenderableRange{base_range, "Base Range", Color{255, 255, 255, 255}}};
  for(auto& entry : action_ranges) ranges.push_back(entry.second);
  viewer_p->render(ranges);
}

PokerRange build_action_range(const PokerRange& base_range, const Action& a, const PokerState& state, const Board& board,
    const DecisionAlgorithm& decision) {
  PokerRange rel_range;
  for(const auto& hand : base_range.hands()) {
    rel_range.add_hand(hand, decision.frequency(a, state, board, hand));
  }
  return rel_range;
}

void update_ranges(std::vector<PokerRange>& ranges, Action a, const PokerState& state, const Board& board, 
    const DecisionAlgorithm& decision) {
  auto action_range = build_action_range(ranges[state.get_active()], a, state, board, decision);
  ranges[state.get_active()] *= action_range;
}

std::vector<PokerRange> build_ranges(const std::vector<Action>& actions, const Board& board, const Strategy<float>& strat) {
  PokerState curr_state = strat.get_config().init_state;
  std::vector<PokerRange> ranges = strat.get_config().init_ranges;
  StrategyDecision decision{strat.get_strategy(), strat.get_config().action_profile};
  for(int aidx = 0; aidx < actions.size(); ++aidx) {
    update_ranges(ranges, actions[aidx], curr_state, board, decision);
    if(aidx != actions.size() - 1) {
      curr_state = curr_state.apply(actions[aidx]); // don't apply last action to avoid incrementing the round before card removal
    }
  }
  for(auto& r : ranges) {
    r.remove_cards(board.as_vector(n_board_cards(curr_state.get_round())));
  }
  return ranges;
}

}
