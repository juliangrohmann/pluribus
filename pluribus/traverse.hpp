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
#include <pluribus/util.hpp>
#include <pluribus/blueprint.hpp>

namespace pluribus {

void traverse_trainer(RangeViewer* viewer_p, const std::string& bp_fn);
void traverse_blueprint(RangeViewer* viewer_p, const std::string& bp_fn);
Action str_to_action(const std::string& str);
void render_ranges(RangeViewer* viewer_p, const PokerRange& base_range, const std::unordered_map<Action, RenderableRange>& action_ranges);

template <class T>
PokerRange build_action_range(const PokerRange& base_range, const Action& a, const PokerState& state, const Board& board, const StrategyStorage<T>& strat, const ActionProfile& profile) {
  auto actions = valid_actions(state, profile);
  PokerRange rel_range;
  for(const auto& hand : base_range.hands()) {
    int cluster = FlatClusterMap::get_instance()->cluster(state.get_round(), board, hand);
    int base_idx = strat.index(state, cluster);
    auto freq = calculate_strategy(strat, base_idx, actions.size());
    int a_idx = std::distance(actions.begin(), std::find(actions.begin(), actions.end(), a));
    rel_range.add_hand(hand, freq[a_idx]);
  }
  return rel_range;
}

template <class T>
std::unordered_map<Action, RenderableRange> build_renderable_ranges(const StrategyStorage<T>& strat, const ActionProfile& profile, 
                                                                    const PokerState& state, const Board& board, PokerRange& base_range) {
  std::unordered_map<Action, RenderableRange> ranges;
  auto actions = valid_actions(state, profile);
  auto color_map = map_colors(actions);
  base_range.remove_cards(board.as_vector(n_board_cards(state.get_round())));
  for(Action a : actions) {
    PokerRange action_range = build_action_range(base_range, a, state, board, strat, profile);
    ranges.insert({a, RenderableRange{base_range * action_range, a.to_string(), color_map[a], true}});
  }
  return ranges;
}

template <class T>
void traverse(RangeViewer* viewer_p, const Strategy<T>& bp) {
  std::string input;
  std::cout << "Board cards: ";
  auto board_cards = bp.get_config().init_board;
  for(uint8_t card : board_cards) std::cout << idx_to_card(card);
  if(board_cards.size() < 5) {
    std::getline(std::cin, input);
    uint8_t missing_cards[5];
    str_to_cards(input, missing_cards);
    for(int i = 0; i < 5 - bp.get_config().init_board.size(); ++i) {
      board_cards.push_back(missing_cards[i]);
    }
  }
  Board board(board_cards);
  std::cout << "Board: " << board.to_string() << "\n";

  PokerState state = bp.get_config().init_state;
  std::vector<PokerRange> ranges = bp.get_config().init_ranges;
  for(int i = 0; i < bp.get_config().poker.n_players; ++i) ranges.push_back(PokerRange::full());
  auto action_ranges = build_renderable_ranges(bp.get_strategy(), bp.get_config().action_profile, state, board, ranges[state.get_active()]);
  render_ranges(viewer_p, ranges[state.get_active()], action_ranges);

  std::cout << state.to_string();
  std::cout << "\nAction: ";
  while(std::getline(std::cin, input)) {
    if(input == "quit") {
      std::cout << "Exiting...\n\n";
      break;
    }
    else if(input == "reset") {
      std::cout << "Resetting...\n\n";
      ranges = bp.get_config().init_ranges;
      state = bp.get_config().init_state;
    }
    else {
      Action action = str_to_action(input);
      std::cout << "\n" << action.to_string() << "\n\n";
      ranges[state.get_active()] = action_ranges.at(action).get_range();
      state = state.apply(action);
    }
    
    if(state.is_terminal()) {
      ranges = bp.get_config().init_ranges;
      state = bp.get_config().init_state;
    }

    action_ranges = build_renderable_ranges(bp.get_strategy(), bp.get_config().action_profile, state, board, ranges[state.get_active()]);
    render_ranges(viewer_p, ranges[state.get_active()], action_ranges);
    std::cout << state.to_string();
    std::cout << "\nAction: ";
  }
}

}

