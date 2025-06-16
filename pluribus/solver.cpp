#include <pluribus/logging.hpp>
#include <pluribus/solver.hpp>

namespace pluribus {

void Solver::solve(const PokerState& state, const std::vector<uint8_t> board, const std::vector<PokerRange>& ranges, const ActionProfile& profile) {
  Logger::log("================================= Solve ==================================");
  if(board.size() != n_board_cards(state.get_round())) {
    Logger::error("Wrong amount of board cards. Round=" + round_to_str(state.get_round()) + ", Board=" + cards_to_str(board));
  }
  _state = SolverState::SOLVING;
  _solve(state, board, ranges, profile);
  _state = SolverState::SOLVED;
}

float RealTimeMCCFR::frequency(const PokerState& state, const Hand& hand, Action action) const {
  if(!_regrets) Logger::error("Regrets are uninitialized.");
  auto actions = valid_actions(state, _regrets->action_profile());
  auto a_it = std::find(actions.begin(), actions.end(), action);
  if(a_it == actions.end()) Logger::error("Action " + action.to_string() + " is not in the action profile or not valid.");
  int a_idx = std::distance(actions.begin(), a_it);
  return _regrets->get(state, HoleCardIndexer::get_instance()->index(hand), a_idx).load();
}

RealTimeSolver::RealTimeSolver(const std::string& bp_fn, const std::filesystem::path& log_dir) {
  if(!create_dir(log_dir)) throw std::runtime_error("Failed to create log directory at " + log_dir.string());
  _log_file = std::filesystem::path{log_dir} / (date_time_str() + "_solver.log");
  std::cout << "Logging to " << _log_file.string();
  Logger::log("Loading from real time solver's blueprint from " + bp_fn + " ...");
  _bp = std::make_shared<SampledBlueprint>();
  std::ifstream is(bp_fn, std::ios::binary);
  cereal::BinaryInputArchive iarchive(is);
  iarchive(*_bp);
  Logger::log("Blueprint loaded.");
}

void RealTimeSolver::new_game(int hero_pos) {
  Logger::log("================================ New Game ================================");
  Logger::log("Game idx=" + std::to_string(_game_idx++));
  _real_state = _bp->get_config().init_state; // TODO: supply state with real stack sizes and keep seperate bp state with in-abstraction stack sizes
  _root_state = _real_state;
  Logger::log("Hero position: " + pos_to_str(hero_pos, _root_state.get_players().size()));
  Logger::log("Real state/Root state:\n" + _root_state.to_string());
  int init_pos = _root_state.get_players().size() == 2 ? 1 : 2;
  if(_root_state.get_bet_level() > 0 || _root_state.get_round() != 0 || _root_state.get_active() != init_pos) {
    Logger::error("Invalid initial state.");
  }

  _ranges = _bp->get_config().init_ranges;
  std::ostringstream oss;
  oss << "Starting ranges:\n";
  for(int p = 0; p < _ranges.size(); ++p) {
    oss << pos_to_str(p, _real_state.get_players().size()) << ": " << _ranges[p].n_combos() << " combos\n";
  }
  Logger::log(oss.str());

  _live_profile = _bp->get_config().action_profile; // TODO: use more actions in live solver than blueprint?
  Logger::log("Live profile:\n" + _live_profile.to_string());

  _board.clear();
  Logger::log("# Board cards: " + std::to_string(_board.size()));
}

std::string chips_to_str(const PokerState& state, int i) {
  return pos_to_str(i, state.get_players().size()) + " chips = " + std::to_string(state.get_players()[i].get_chips());
}

void RealTimeSolver::update_state(const PokerState& state) {
  Logger::log("============================== Update State ==============================");
  Logger::log(state.to_string());
  if(!state.get_action_history().is_consistent(_real_state.get_action_history())) {
    Logger::log("Real action history: " + _real_state.get_action_history().to_string());
    Logger::log("Updated action history: " + _real_state.get_action_history().to_string());
    Logger::error("Inconsistent action histories.");
  }
  if(state.get_action_history().size() <= _real_state.get_action_history().size()) {
    Logger::error("No new actions in updated state. Real actions=" + std::to_string(_real_state.get_action_history().size()) + 
      ", Updated actions=" + std::to_string(state.get_action_history().size()));
  }

  for(Action a : state.get_action_history().slice(_real_state.get_action_history().size()).get_history()) {
    _apply_action(a);
  }

  for(int p = 0; p < state.get_players().size(); ++p) {
    if(state.get_players()[p].get_betsize() != _real_state.get_players()[p].get_betsize()) {
      Logger::error(pos_to_str(p, _real_state.get_players().size()) + " betsize mismatch. " +
        "Real=" + std::to_string(_real_state.get_players()[p].get_betsize()) + ", "
        "Updated=" + std::to_string(state.get_players()[p].get_betsize()));
    }
    if(state.get_players()[p].get_chips() != _real_state.get_players()[p].get_chips()) {
      Logger::error(pos_to_str(p, _real_state.get_players().size()) + " chips mismatch. " +
        "Real=" + std::to_string(_real_state.get_players()[p].get_chips()) + ", "
        "Updated=" + std::to_string(state.get_players()[p].get_chips()));
    }
  }
  if(state.get_bet_level() != _real_state.get_bet_level()) {
    Logger::error("Bet level mismatch. Real=" + std::to_string(_real_state.get_bet_level()) + 
      ", Updated=" + std::to_string(state.get_bet_level()));
  }
  if(state.get_max_bet() != _real_state.get_max_bet()) {
    Logger::error("Max bet mismatch. Real=" + std::to_string(_real_state.get_max_bet()) + ", Updated=" + std::to_string(state.get_max_bet()));
  }
  if(state.get_pot() != _real_state.get_pot()) {
    Logger::error("Pot mismatch. Real=" + std::to_string(_real_state.get_pot()) + ", Updated=" + std::to_string(state.get_pot()));
  }
}

void RealTimeSolver::update_board(const std::vector<uint8_t> updated_board) {
  Logger::log("============================== Update Board ==============================\n");
  Logger::log("Previous board: " + cards_to_str(_board));
  Logger::log("Updated board: " + cards_to_str(updated_board));
  if(_board.size() <= updated_board.size()) Logger::error("No new cards on updated board.");
  for(int i = 0; i < _board.size(); ++i) {
    if(_board[i] != updated_board[i]) Logger::error("Inconsistent boards.");
  }
  _board = updated_board;
}

Solution RealTimeSolver::solution(const PokerState& state, const Hand& hand) {
  Solution solution;
  solution.actions = valid_actions(state, _live_profile);
  for(Action a : solution.actions) {
    solution.freq.push_back(_solver->frequency(state, hand, a));
  }
  return solution;
}

void RealTimeSolver::_init_solver() {
  Logger::log("Initializing solver: RealTimeMCCFR");
  _solver = std::unique_ptr<Solver>{new RealTimeMCCFR{_bp}};
}

void RealTimeSolver::_apply_action(Action a) {
  Logger::log("Applying action: " + a.to_string());
  _real_state.apply(a);
  if(_real_state.get_round() <= 3 && _real_state.get_round() > _root_state.get_round()) {
    _root_state = _real_state;
    _init_solver();
    _solver->solve(_root_state, _board, _ranges, _live_profile);
  }
}

}