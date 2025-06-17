#include <iostream>
#include <iomanip>
#include <unordered_map>
#include <vector>
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
  auto bp = cereal_load<BlueprintTrainer>(bp_fn);
  std::cout << "Success.\n";
  traverse(viewer_p, bp);
}

void traverse_blueprint(RangeViewer* viewer_p, const std::string& bp_fn) {
  std::cout << "Loading blueprint from " << bp_fn << " for traversal... " << std::flush;
  auto bp = cereal_load<LosslessBlueprint>(bp_fn);
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

std::vector<PokerRange> build_ranges(const std::vector<Action>& actions, const Board& board, const Strategy<float>& strat) {
  PokerState curr_state = strat.get_config().init_state;
  std::vector<PokerRange> ranges = strat.get_config().init_ranges;
  for(int aidx = 0; aidx < actions.size(); ++aidx) {
    auto action_range = build_action_range(ranges[curr_state.get_active()], actions[aidx], curr_state, board, 
                                      strat.get_strategy(), strat.get_config().action_profile);
    for(int ridx = 0; ridx < ranges.size(); ++ridx) {
      if(ridx != curr_state.get_active()) {
        ranges[ridx] = ranges[ridx].bayesian_update(ranges[curr_state.get_active()], action_range);
      }
    }
    ranges[curr_state.get_active()] *= action_range;
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
