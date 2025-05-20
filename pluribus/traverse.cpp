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
  auto bp = cereal_load<Blueprint>(bp_fn);
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

}
