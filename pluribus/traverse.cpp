#include <iostream>
#include <unordered_map>
#include <vector>
#include <pluribus/range.hpp>
#include <pluribus/mccfr.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/range_viewer.hpp>
#include <pluribus/util.hpp>
#include <pluribus/traverse.hpp>

namespace pluribus {

void traverse(RangeViewer* viewer_p, const DecisionAlgorithm& decision, const SolverConfig& config) {
  std::string input;
  std::cout << "Board cards: ";
  auto board_cards = config.init_board;
  for(uint8_t card : board_cards) std::cout << idx_to_card(card);
  if(board_cards.size() < 5) {
    std::getline(std::cin, input);
    uint8_t missing_cards[5];
    str_to_cards(input, missing_cards);
    for(int i = 0; i < 5 - config.init_board.size(); ++i) {
      board_cards.push_back(missing_cards[i]);
    }
  }
  Board board(board_cards);
  std::cout << "Board: " << board.to_string() << "\n";

  PokerState state = config.init_state;
  std::vector<PokerRange> ranges = config.init_ranges;
  for(int i = 0; i < config.poker.n_players; ++i) ranges.push_back(PokerRange::full());
  auto action_ranges = build_renderable_ranges(decision, config.action_profile, state, board, ranges[state.get_active()]);
  render_ranges(viewer_p, ranges[state.get_active()], action_ranges);

  std::cout << state.to_string();
  std::cout << "\nAction: ";
  while(std::getline(std::cin, input)) {
    if(input == "quit") {
      std::cout << "Exiting...\n\n";
      break;
    }
    if(input == "reset") {
      std::cout << "Resetting...\n\n";
      ranges = config.init_ranges;
      state = config.init_state;
    }
    else {
      Action action = str_to_action(input);
      std::cout << "\n" << action.to_string() << "\n\n";
      ranges[state.get_active()] = action_ranges.at(action).get_range();
      state = state.apply(action);
    }

    if(state.is_terminal()) {
      ranges = config.init_ranges;
      state = config.init_state;
    }

    action_ranges = build_renderable_ranges(decision, config.action_profile, state, board, ranges[state.get_active()]);
    render_ranges(viewer_p, ranges[state.get_active()], action_ranges);
    std::cout << state.to_string();
    std::cout << "\nAction: ";
  }
}

void traverse_tree(RangeViewer* viewer_p, const std::string& bp_fn) {
  std::cout << "Loading tree blueprint solver from " << bp_fn << " for traversal... " << std::flush;
  TreeBlueprintSolver bp;
  cereal_load(bp, bp_fn);
  std::cout << "Success.\n";
  traverse(viewer_p, TreeDecision{bp.get_strategy(), bp.get_config().init_state}, bp.get_config());
}

void traverse_blueprint(RangeViewer* viewer_p, const std::string& bp_fn) {
  std::cout << "Loading blueprint from " << bp_fn << " for traversal... " << std::flush;
  LosslessBlueprint bp;
  cereal_load(bp, bp_fn);
  std::cout << "Success.\n";
  traverse(viewer_p, TreeDecision{bp.get_strategy(), bp.get_config().init_state}, bp.get_config());
}

Action str_to_action(const std::string& str) {
  if(str.starts_with("check") || str.starts_with("call")) return Action::CHECK_CALL;
  if(str.starts_with("fold")) return Action::FOLD;
  if(str.starts_with("all-in")) return Action::ALL_IN;
  if(str.starts_with("bet")) return Action(std::stoi(str.substr(4)) / 100.0);
  throw std::runtime_error("Invalid action string: " + str);
}

void render_ranges(RangeViewer* viewer_p, const PokerRange& base_range, const std::unordered_map<Action, RenderableRange>& action_ranges) {
  std::vector ranges{RenderableRange{base_range, "Base Range", Color{255, 255, 255, 255}}};
  for(auto& r : action_ranges | std::views::values) ranges.push_back(r);
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

void update_ranges(std::vector<PokerRange>& ranges, const Action a, const PokerState& state, const Board& board,
    const DecisionAlgorithm& decision) {
  const auto action_range = build_action_range(ranges[state.get_active()], a, state, board, decision);
  ranges[state.get_active()] *= action_range;
}

std::vector<PokerRange> build_ranges(const std::vector<Action>& actions, const Board& board, const Strategy<float>& strat) {
  PokerState curr_state = strat.get_config().init_state;
  std::vector<PokerRange> ranges = strat.get_config().init_ranges;
  const TreeDecision decision{strat.get_strategy(), strat.get_config().init_state};
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

std::unordered_map<Action, RenderableRange> build_renderable_ranges(const DecisionAlgorithm& decision, const ActionProfile& profile, 
    const PokerState& state, const Board& board, PokerRange& base_range) {
  std::unordered_map<Action, RenderableRange> ranges;
  const auto actions = valid_actions(state, profile);
  auto color_map = map_colors(actions);
  base_range.remove_cards(board.as_vector(n_board_cards(state.get_round())));
  for(Action a : actions) {
    PokerRange action_range = build_action_range(base_range, a, state, board, decision);
    ranges.insert({a, RenderableRange{base_range * action_range, a.to_string(), color_map[a], true}});
  }
  return ranges;
}

}
