#include <pluribus/logging.hpp>
#include <pluribus/traverse.hpp>
#include <pluribus/pluribus.hpp>

namespace pluribus {

class RealTimeDecision : public DecisionAlgorithm {
public:
  RealTimeDecision(const LosslessBlueprint& preflop_bp, const std::shared_ptr<const Solver>& solver)
      : _preflop_decision{StrategyDecision{preflop_bp.get_strategy(), preflop_bp.get_config().action_profile}}, _solver{solver} {}

  float frequency(const Action a, const PokerState& state, const Board& board, const Hand& hand) const override {
    if(_solver) return _solver->frequency(a, state, board, hand);
    if(state.get_round() == 0) return _preflop_decision.frequency(a, state, board, hand);
    Logger::error("Cannot decide postflop frequency without solver.");
  }

private:
  const StrategyDecision<float> _preflop_decision;
  const std::shared_ptr<const Solver> _solver;
};

Pluribus::Pluribus(const std::shared_ptr<const LosslessBlueprint>& preflop_bp, const std::shared_ptr<const SampledBlueprint>& sampled_bp)
    : _preflop_bp{preflop_bp}, _sampled_bp{sampled_bp} {}

void Pluribus::new_game(const std::vector<std::string>& players) {
  Logger::log("================================ New Game ================================");
  std::ostringstream oss;
  Logger::log("Players: " + join_strs(players, ", "));

  _real_state = _sampled_bp->get_config().init_state; // TODO: supply state with real stack sizes
  _root_state = _real_state;
  _mapped_actions = _real_state.get_action_history();

  if(players.size() != _real_state.get_players().size()) {
    Logger::error("Player number mismatch. Expected " + std::to_string(_real_state.get_players().size()) + " players.");
  }
  Logger::log("Real state/Root state:\n" + _root_state.to_string());
  if(const int init_pos = _root_state.get_players().size() == 2 ? 1 : 2;
      _root_state.get_bet_level() > 0 || _root_state.get_round() != 0 || _root_state.get_active() != init_pos) {
    Logger::error("Invalid initial state.");
  }

  _ranges = _sampled_bp->get_config().init_ranges;
  oss << "Starting ranges:\n";
  for(int p = 0; p < _ranges.size(); ++p) {
    oss << pos_to_str(p, _real_state.get_players().size()) << ": " << _ranges[p].n_combos() << " combos\n";
  }
  Logger::log(oss.str());

  _live_profile = _sampled_bp->get_config().action_profile; // TODO: use more actions in live solver than blueprint?
  Logger::log("Live profile:\n" + _live_profile.to_string());

  _board.clear();
  Logger::log("# Board cards: " + std::to_string(_board.size()));
}

std::string chips_to_str(const PokerState& state, const int i) {
  return pos_to_str(i, state.get_players().size()) + " chips = " + std::to_string(state.get_players()[i].get_chips());
}

void Pluribus::update_state(const Action action, const int pos) {
  Logger::log("============================== Update State ==============================");
  Logger::log(pos_to_str(pos, _real_state.get_players().size()) + ": " + action.to_string());
  if(_real_state.get_active() != pos) {
    Logger::error("Wrong player is acting. Expected " + pos_to_str(_real_state.get_active(), _real_state.get_players().size()) + " to act.");
  }
  _apply_action(action);
}

void Pluribus::update_board(const std::vector<uint8_t>& updated_board) {
  Logger::log("============================== Update Board ==============================\n");
  Logger::log("Previous board: " + cards_to_str(_board));
  Logger::log("Updated board: " + cards_to_str(updated_board));
  if(_board.size() <= updated_board.size()) Logger::error("No new cards on updated board.");
  for(int i = 0; i < _board.size(); ++i) {
    if(_board[i] != updated_board[i]) Logger::error("Inconsistent boards.");
  }
  _board = updated_board;
}

Solution Pluribus::solution(const PokerState& state, const Hand& hand) const {
  Solution solution;
  solution.actions = valid_actions(state, _live_profile);
  const RealTimeDecision decision{*_preflop_bp, _solver};
  for(const Action a : solution.actions) {
    solution.freq.push_back(decision.frequency(a, state, Board{_board}, hand));
  }
  return solution;
}

int terminal_round(const PokerState& root) { 
  return root.get_round() >= 2 || (root.get_round() == 1 && root.active_players() == 2) ? 4 : root.get_round() + 1;
}

void Pluribus::_init_solver() {
  Logger::log("Initializing solver: MappedRealTimeSolver");
  SolverConfig config{_sampled_bp->get_config().poker};
  config.init_state = _root_state;
  config.init_board = _board;
  config.init_ranges = _ranges;
  config.action_profile = _live_profile;
  RealTimeSolverConfig rt_config;
  rt_config.terminal_round = terminal_round(_root_state);
  rt_config.terminal_bet_level = 999;
  _solver = std::unique_ptr<Solver>{new MappedRealTimeSolver{_sampled_bp, rt_config}};
}

bool can_solve(const PokerState& root) {
  return root.get_round() > 0 || root.active_players() <= 4;
}

bool should_solve(const PokerState& root) {
  return can_solve(root) && root.get_round() > 0;
}

bool is_off_tree(Action a, const PokerState& state, const ActionProfile& profile) {
  // TODO
  return false;
}

void Pluribus::_apply_action(const Action a) {
  Logger::log("Applying action: " + a.to_string());
  _real_state = _real_state.apply(a);
  if(!can_solve(_root_state) && can_solve(_real_state)) {
    _update_root();
  }
  if(can_solve(_root_state) && is_off_tree(a, _real_state, _live_profile)) {
    // TODO: interrupt if solving, add action, re-solve
  }
  else {
    Action mapped = a; // TODO: map preflop actions
    _mapped_actions.push_back(a);
  }
}

void Pluribus::_update_root() {
  PokerState curr_state = _root_state;
  const RealTimeDecision decision{*_preflop_bp, _solver};
  std::ostringstream oss;
  for(Action a : _real_state.get_action_history().slice(_root_state.get_action_history().size()).get_history()) {
    oss << pos_to_str(curr_state.get_active(), _ranges.size()) << " action applied to root: " + a.to_string() << ", combos: "
        << std::fixed << std::setprecision(2) << _ranges[curr_state.get_active()].n_combos();
    update_ranges(_ranges, a, curr_state, Board{_board}, decision);
    oss << " -> " << _ranges[curr_state.get_active()].n_combos();
    Logger::dump(oss);
    curr_state = curr_state.apply(a);
  }
  for(int i = 0; i < _ranges.size(); ++i) {
    oss << pos_to_str(i, _ranges.size()) << " card removal, combos: " << _ranges[i].n_combos();
    _ranges[i].remove_cards(_board);
    oss << " -> " << _ranges[i].n_combos();
    Logger::dump(oss);
  }

  if(curr_state != _real_state) {
    Logger::error("Updated root state does not match real state.\nUpdated root:\n" + 
        curr_state.to_string() + "\nReal state:\n" + _real_state.to_string());
  }
  _root_state = _real_state;
  Logger::log("New root:\n" + _root_state.to_string());
  if(should_solve(_root_state)) {
    Logger::log("Should solve.");
    // TODO: interrupt if solving
    _init_solver();
    _solver->solve(100'000'000'000L);
  }
}

}