#include <pluribus/logging.hpp>
#include <pluribus/pluribus.hpp>
#include <pluribus/translate.hpp>
#include <pluribus/traverse.hpp>

namespace pluribus {

std::string Solution::to_string() const {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(4) << "Solution: actions=" << actions_to_str(actions) << ", freq=[";
  for(int i = 0; i < actions.size(); ++i) oss << freq[i] << (i == freq.size() - 1 ? "]" : ", ");
  oss << ", aligned=" << (aligned ? "true" : "false") << "\n";
  return oss.str();
}

std::string FrozenNode::to_string() const {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(4) << "FrozenNode: freq=[";
  for(int i = 0; i < freq.size(); ++i) oss << freq[i] << (i == freq.size() - 1 ? "]" : ", ");
  oss << ", actions=[" << actions_to_str(actions) << "], hand=" << hand.to_string() << ", board=" << cards_to_str(board)
      << ", live_actions=" << live_actions.to_string();
  return oss.str();
}

class RealTimeDecision : public DecisionAlgorithm {
public:
  RealTimeDecision(const LosslessBlueprint& preflop_bp, const std::shared_ptr<const Solver>& solver, const std::vector<FrozenNode>& frozen)
      : _preflop_decision{TreeDecision{preflop_bp.get_strategy(), preflop_bp.get_config().init_state, false}}, _solver{solver}, _frozen{frozen} {}

  float frequency(const Action a, const PokerState& state, const Board& board, const Hand& hand) const override {
    if(_solver) return _solver->frequency(a, state, board, hand);
    if(state.get_round() > 0) Logger::error("Cannot decide postflop frequency without solver.");
    for(const auto& node : _frozen) {
      if(hand == node.hand && state.get_action_history() == node.live_actions) {
        return node.freq[index_of(a, node.actions)];
      }
    }
    return _preflop_decision.frequency(a, state, board, hand);
  }

private:
  const TreeDecision<float> _preflop_decision;
  const std::shared_ptr<const Solver> _solver;
  std::vector<FrozenNode> _frozen;
};

Pluribus::Pluribus(const std::array<ActionProfile, 4>& live_profiles, const std::shared_ptr<const LosslessBlueprint>& preflop_bp,
  const std::shared_ptr<const SampledBlueprint>& sampled_bp)
    : _preflop_bp{preflop_bp}, _sampled_bp{sampled_bp}, _init_profiles{live_profiles} {
  Logger::log("Pluribus action profile:\n" + _sampled_bp->get_config().action_profile.to_string());
  Logger::log((HoleCardIndexer::get_instance() ? "Initialized" : "Failed to initialize") + std::string{" hole card indexer."});
  Logger::log((HandIndexer::get_instance() ? "Initialized" : "Failed to initialize") + std::string{" hand indexer."});
  Logger::log((BlueprintClusterMap::get_instance() ? "Initialized" : "Failed to initialize") + std::string{" blueprint cluster map."});
  Logger::log((RealTimeClusterMap::get_instance() ? "Initialized" : "Failed to initialize") + std::string{" real time cluster map."});
  _start_worker();
}

Pluribus::~Pluribus() {
  {
    std::lock_guard lk(_solver_mtx);
    _running_worker = false;
    _pending_job.reset();
    if(_solver) _solver->interrupt();
  }
  _solver_cv.notify_all();
  if(_solver_thread.joinable()) _solver_thread.join();
}

void Pluribus::new_game(const std::vector<int>& stacks, const Hand& hero_hand, const int hero_pos) {
  Logger::log("================================ New Game ================================");
  valid = true;
  std::ostringstream oss;
  Logger::log("Stacks: " + join_as_strs(stacks, ", "));
  const auto& poker_config = _sampled_bp->get_config().poker;
  if(hero_pos < 0 || hero_pos >= poker_config.n_players) {
    Logger::error("0 <= Hero position < " + std::to_string(poker_config.n_players) + " required. Hero position=" + std::to_string(hero_pos));
  }
  _hero_hand = hero_hand;
  Logger::log("Hero hand: " + _hero_hand.to_string());
  _hero_pos = hero_pos;
  Logger::log("Hero position: " + std::to_string(_hero_pos) + " (" + pos_to_str(_hero_pos, poker_config.n_players, poker_config.straddle) + ")");
  if(stacks.size() != poker_config.n_players) {
    Logger::error("Player number mismatch. Expected " + std::to_string(poker_config.n_players) + " players.");
  }

  {
    std::lock_guard lk(_solver_mtx);
    if(_solver) _solver->interrupt();
    _solver = nullptr;
  }
  _real_state = PokerState{poker_config.n_players, stacks, poker_config.ante, poker_config.straddle};
  _root_state = _real_state;
  _mapped_bp_actions = ActionHistory{};
  _mapped_live_actions = ActionHistory{};
  _live_profile = _init_profiles[_real_state.get_round()];
  _frozen = std::vector<FrozenNode>{};

  Logger::log("Real state/Root state:\n" + _root_state.to_string());
  const int init_pos = _root_state.get_players().size() == 2 ? 1 : 2;
  if(_root_state.get_bet_level() > 1 || _root_state.get_round() != 0 || _root_state.get_active() != init_pos) {
    Logger::error("Invalid initial state.");
  }

  _ranges = _sampled_bp->get_config().init_ranges;
  oss << "Starting ranges:\n";
  for(int p = 0; p < _ranges.size(); ++p) {
    oss << pos_to_str(p, _real_state.get_players().size(), _real_state.is_straddle()) << ": " << _ranges[p].n_combos() << " combos\n";
  }
  Logger::log(oss.str());

  _board = std::vector<uint8_t>{};
  Logger::log("# Board cards: " + std::to_string(_board.size()));
}

std::string chips_to_str(const PokerState& state, const int i) {
  return pos_to_str(i, state.get_players().size(), state.is_straddle()) + " chips = " + std::to_string(state.get_players()[i].get_chips());
}

void handle_misaligned_state_update(PokerState& real_state, const PokerState& mapped_state, const Action action) {
  // TODO: replace closest actions in profile with exact actions instead and resolve when misaligned
  Logger::log("WARNING: Mapped state is NOT aligned. Not mapping until aligned.");
  Logger::log("Real state:\n" + real_state.to_string());
  Logger::log("Mapped live state:\n" + mapped_state.to_string());
  real_state.apply_in_place(action);
}

void Pluribus::update_state(const Action action, const int pos) {
  try {
    Logger::log("============================== Update State ==============================");
    Logger::log(pos_to_str(pos, _real_state.get_players().size(), _real_state.is_straddle()) + ": " + action.to_string());
    if(_real_state.get_active() != pos) {
      Logger::error("Wrong player is acting. Expected " + pos_to_str(_real_state) + " to act.");
    }
    const PokerState mapped_state = _root_state.apply(_mapped_live_actions);
    if(mapped_state.get_round() != _real_state.get_round() || mapped_state.get_active() != _real_state.get_active()) {
      handle_misaligned_state_update(_real_state, mapped_state, action);
    }
    else if(const int n_cards = n_board_cards(mapped_state.get_round()); n_cards > _board.size()) {
      Logger::error("Expected board update. Expected cards=" + std::to_string(n_cards) + ", Board=" + cards_to_str(_board));
    }
    else {
      _apply_action(action, {});
    }
  }
  catch(const std::exception& e) {
    _set_invalid(e);
  }
}

void Pluribus::hero_action(const Action action, const std::vector<float>& freq) {
  try {
    Logger::log("============================== Hero Action ===============================");
    Logger::log(pos_to_str(_hero_pos, _real_state.get_players().size(), _real_state.is_straddle()) + " (Hero): " + action.to_string());
    if(_real_state.get_active() != _hero_pos) {
      Logger::error("Wrong player is acting. Expected " + pos_to_str(_real_state) + " (hero) to act.");
    }
    const PokerState mapped_state = _root_state.apply(_mapped_live_actions);
    if(mapped_state.get_round() != _real_state.get_round() || mapped_state.get_active() != _real_state.get_active()) {
      handle_misaligned_state_update(_real_state, mapped_state, action);
    }
    else {
      _apply_action(action, freq);
    }
  }
  catch(const std::exception& e) {
    _set_invalid(e);
  }
}

void Pluribus::update_board(const std::vector<uint8_t>& updated_board) {
  try {
    Logger::log("============================== Update Board ==============================");
    Logger::log("Previous board: " + cards_to_str(_board));
    Logger::log("Updated board: " + cards_to_str(updated_board));
    if(_board.size() >= updated_board.size()) Logger::error("No new cards on updated board.");
    for(int i = 0; i < _board.size(); ++i) {
      if(_board[i] != updated_board[i]) Logger::error("Inconsistent boards.");
    }
    _board = updated_board;
    if(_real_state.get_round() > _root_state.get_round() && _can_solve(_real_state)) {
      Logger::log("Street advanced. Updating root...");
      _update_root(true);
    }
  }
  catch(const std::exception& e) {
    _set_invalid(e);
  }
}

Solution Pluribus::solution(const Hand& hand) {
  try {
    Logger::log("================================ Solution ================================");
    Solution solution;
    const PokerState mapped_state = _root_state.apply(_mapped_live_actions);
    if(mapped_state.get_round() == _real_state.get_round() && mapped_state.get_active() == _real_state.get_active()) {
      Logger::log("Mapped state is aligned. Returning real solution.");
      solution.aligned = true;
      {
        std::lock_guard lk(_solver_mtx);
        if(_solver && _solver->get_real_time_config().is_state_terminal(mapped_state)) {
          Logger::error("Requested solution extends past the end of the non-terminal solve. "
              + _solver->get_real_time_config().to_string() + ", Terminal state:\n" + mapped_state.to_string());
        }
        solution.actions = _get_solution_actions();
        const RealTimeDecision decision{*_preflop_bp, _solver, _frozen};
        for(const Action a : solution.actions) {
          solution.freq.push_back(decision.frequency(a, mapped_state, Board{_board}, hand));
        }
      }
    }
    else {
      // TODO: replace closest actions in profile with exact actions instead and resolve when misaligned
      Logger::log("WARNING: Mapped state is NOT aligned. Returning check/call until aligned.");
      Logger::log("Real state:\n" + _real_state.to_string());
      Logger::log("Mapped live state:\n" + mapped_state.to_string());
      solution.actions = {Action::CHECK_CALL};
      solution.freq = {1.0};
      solution.aligned = false;
    }
    Logger::log(solution.to_string());
    return solution;
  }
  catch(const std::exception& e) {
    _set_invalid(e);
    return Solution{};
  }
}

void Pluribus::save_range(const std::string& fn) {
  try {
    Logger::log("=============================== Save Range ===============================");
    PngRangeViewer viewer{fn};
    const PokerState mapped_state = _root_state.apply(_mapped_live_actions);
    {
      std::lock_guard lk(_solver_mtx);
      if(_solver && _solver->get_real_time_config().is_state_terminal(mapped_state)) {
        Logger::error("Requested range extends past the end of the non-terminal solve. "
            + _solver->get_real_time_config().to_string() + ", Terminal state:\n" + mapped_state.to_string());
      }
      const RealTimeDecision decision{*_preflop_bp, _solver, _frozen};
      auto live_ranges = _ranges;
      PokerState curr_state = _root_state;
      for(Action a : _mapped_live_actions.get_history()) {
        std::cout << "Updating range: " << a.to_string() << "\n";
        update_ranges(live_ranges, a, curr_state, Board{_board}, decision);
        curr_state.apply_in_place(a);
      }
      std::cout << "Building renderable ranges...\n";
      const std::vector<Action> actions = _get_solution_actions();
      const auto action_ranges = build_renderable_ranges(decision, actions, mapped_state, Board{_board}, live_ranges[mapped_state.get_active()]);
      std::cout << "Rendering ranges...\n";
      render_ranges(&viewer, live_ranges[mapped_state.get_active()], action_ranges);
    }
  }
  catch(const std::exception& e) {
    _set_invalid(e);
  }
}

std::vector<Action> Pluribus::_get_solution_actions() const {
  std::vector<Action> actions;
  if(_solver) {
    Logger::log("Getting solution actions...");
    Logger::log("Applying live actions to solver. Mapped live actions: " + _mapped_live_actions.to_string());
    actions = _solver->get_strategy()->apply(_mapped_live_actions.get_history())->get_value_actions();
  }
  else {
    if(!_mapped_bp_actions.get_history().empty()) Logger::error("No solver available, but mapped blueprint actions exist.");
    Logger::log("Applying live actions to preflop blueprint. Mapped live actions: " + _mapped_live_actions.to_string());
    actions = _preflop_bp->get_strategy()->apply(_mapped_live_actions.get_history())->get_value_actions();
  }
  Logger::log("Value actions=" + actions_to_str(actions));
  return actions;
}

int terminal_round(const PokerState& root) { 
  return root.get_round() >= 2 || (root.get_round() == 1 && root.active_players() == 2) ? 4 : root.get_round() + 1;
}

int terminal_bet_level(const PokerState& root) {
  if(root.get_round() == 1 && root.active_players() > 2) {
    return root.get_bet_level() + 2;
  }
  if(root.get_round() == 0 && root.active_players() > 4) {
    return root.get_bet_level() + 2;
  }
  return 999;
}

void Pluribus::_enqueue_job(const bool force_terminal) {
  Logger::log("Initializing solve job...");
  SolverConfig config{_sampled_bp->get_config().poker, _live_profile};
  config.rake = _sampled_bp->get_config().rake;
  config.init_state = _root_state;
  config.init_board = _board;
  config.init_ranges = _ranges;
  RealTimeSolverConfig rt_config;
  rt_config.bias_profile = BiasActionProfile{};
  rt_config.init_actions = _mapped_bp_actions.get_history();
  rt_config.terminal_round = force_terminal ? 4 : terminal_round(_root_state);
  rt_config.terminal_bet_level = force_terminal ? 999 : terminal_bet_level(_root_state);
  SolveJob job{config, rt_config};
  const auto ack = job.ack.get_future();
  {
    std::lock_guard lk(_solver_mtx);
    if(_solver) _solver->interrupt();
    _pending_job = std::move(job);
  }
  Logger::log("Enqueued job.");
  _solver_cv.notify_one();
  ack.wait();
}

bool is_off_tree(const Action a, const std::vector<Action>& actions, const PokerState& state) {
  if(a.get_bet_type() <= 0.0f) return false;
  float min_diff = 100.0f;
  Action closest = Action::UNDEFINED;
  for(const Action action : actions) {
    if(action.get_bet_type() > 0.0f || action == Action::ALL_IN) {
      const float frac = action == Action::ALL_IN ? fractional_bet_size(state, total_bet_size(state, action)) : action.get_bet_type();
      const float diff = abs(frac - a.get_bet_type());
      if(diff < min_diff) {
        min_diff = diff;
        closest = action;
      }
    }
  }
  if(closest == Action::UNDEFINED) {
    Logger::error("Unexpected actions during off-tree check: Action=" + a.to_string() + ", Tree actions=" + actions_to_str(actions));
  }
  const int action_size = total_bet_size(state, a);
  const int closest_size = total_bet_size(state, closest);
  const int total_diff = abs(action_size - closest_size);
  if(min_diff > 0.25 && total_diff > 150) {
    Logger::log("Action is off tree: Action=" + a.to_string() + ", Tree actions=" + actions_to_str(actions));
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << "Max frac difference=" << min_diff << ", Action size=" << std::setprecision(0) << action_size
        << ", Closest size=" << closest_size << ", Max total difference=" << total_diff;
    Logger::dump(oss);
    return true;
  }
  return false;
}

void Pluribus::_apply_action(const Action a, const std::vector<float>& freq) {
  const PokerState prev_real_state = _real_state;
  std::vector<Action> actions;
  {
    std::lock_guard lk(_solver_mtx);
    Logger::log("Applying action: " + a.to_string());
    actions = _get_solution_actions();
    if(prev_real_state.get_active() == _hero_pos) {
      const FrozenNode frozen_node{actions, freq, _hero_hand, _board, _mapped_live_actions};
      Logger::log("New frozen node: " + frozen_node.to_string());
      _frozen.push_back(frozen_node);
      if(freq.size() != actions.size()) {
        Logger::error("Freeze frequency amount mismatch:\nActions=" + actions_to_str(actions));
      }
      if(_solver) {
        Logger::log("Applying frozen node to current solver...");
        _solver->freeze(frozen_node.freq, frozen_node.hand, Board{frozen_node.board}, frozen_node.live_actions);
      }
    }
  }

  bool should_solve = false;
  if(is_off_tree(a, actions, prev_real_state)) {
    if(prev_real_state.get_active() == _hero_pos) {
      Logger::log("WARNING: Hero is off tree.");
    }
    else {
      should_solve = true;
      Logger::log("Action is off-tree. Adding to live actions...");
      _live_profile.add_action(a, prev_real_state);
      Logger::log("New live profile:\n" + _live_profile.to_string());
      actions = valid_actions(prev_real_state, _live_profile);
      // TODO: remap actions to new live profile
      // TODO: if root is updated afterwards before the solve can start, seg faults during root update
      // TODO: if root is updated afterwards before the solve can converge, assigns bad ranges during root update
      // Logger::log("Remapping live actions...");
      // Logger::log("Bef mapped live actions: " + _mapped_live_actions.to_string());
      // _mapped_live_actions = ActionHistory{};
      // PokerState curr_state = _root_state;
      // for(Action real_action : _real_state.get_action_history().slice(_root_state.get_action_history().size()).get_history()) {
      //   const Action translated = translate_pseudo_harmonic(a, valid_actions(curr_state, _live_profile), prev_real_state);
      //   _mapped_live_actions.push_back(translated);
      //   Logger::log(real_action.to_string() + " -> " + translated.to_string());
      //   curr_state = curr_state.apply(real_action);
      // }
      // Logger::log("New mapped live actions: " + _mapped_live_actions.to_string());

      // don't need to remap if hero is never off tree
      // Logger::log("Remapping frozen nodes to new live profile...");
      // for(auto& node : _frozen) {
      //   std::ostringstream oss;
      //   oss << "    \n" << node.to_string() << "\n -> ";
      //   PokerState remapped_state = _root_state.apply(node.live_actions);
      //   auto remapped_actions = valid_actions(remapped_state, _live_profile);
      //   auto remapped_freq = std::vector<float>(remapped_actions.size());
      //   for(int i = 0; i < remapped_actions.size(); ++i) {
      //     auto it = std::find(node.actions.begin(), node.actions.end(), remapped_actions[i]);
      //     if(it != node.actions.end()) {
      //       remapped_freq[i] = node.freq[std::distance(node.actions.begin(), it)];
      //     }
      //     else {
      //       remapped_freq[i] = 0.0;
      //     }
      //   }
      //   node.actions = remapped_actions;
      //   node.freq = remapped_freq;
      //   oss << node.to_string();
      //   Logger::dump(oss);
      // }
    }
  }

  _real_state = _real_state.apply(a);
  Logger::log("New state:\n" + _real_state.to_string());
  const Action translated = translate_pseudo_harmonic(a, actions, prev_real_state);
  _mapped_live_actions.push_back(translated);
  Logger::log("Live action translation: " + a.to_string() + " -> " + translated.to_string());

  // TODO: don't update root if current solve is terminal? how fast do future streets converge in terminal solves?
  if(_real_state.get_round() > _root_state.get_round() && _can_solve(_real_state)) {
    Logger::log("Round advanced. Updating root...");
    _update_root(true);
  }
  else if(_solver && _real_state.get_bet_level() >= _solver->get_real_time_config().terminal_bet_level && _can_solve(_real_state)) {
    Logger::log("Bet level advanced. Updating root...");
    _update_root(true);
  }
  else if(should_solve) {
    _enqueue_job(false);
  }
  else if(!_can_solve(prev_real_state)) {
    Logger::log("Root is not solvable yet. Updating root...");
    _update_root(false);
  }
}

void Pluribus::_update_root(const bool solve) {
  Logger::log("Updating root...");
  Logger::log("Root state:\n" + _root_state.to_string());
  Logger::log("Real state:\n" + _real_state.to_string());
  Logger::log("Mapped blueprint actions=" + _mapped_bp_actions.to_string());
  Logger::log("Mapped live actions=" + _mapped_live_actions.to_string());
  std::ostringstream oss;
  PokerState curr_state = _root_state;
  PokerState bp_state = _root_state;
  PokerState live_state = _root_state;
  bool force_terminal = false;
  {
    std::lock_guard lk(_solver_mtx);
    const RealTimeDecision decision{*_preflop_bp, _solver, _frozen};
    const TreeStorageNode<uint8_t>* bp_node = _sampled_bp->get_strategy()->apply(_mapped_bp_actions.get_history());
    std::vector<Action> real_history = _real_state.get_action_history().slice(_root_state.get_action_history().size()).get_history();
    for(int h_idx = 0; h_idx < _mapped_live_actions.size(); ++h_idx) {
      Logger::log("Processing next live action: " + _mapped_live_actions.get(h_idx).to_string());
      Action real_action = real_history.size() > h_idx ? real_history[h_idx] : Action::UNDEFINED;

      if(!force_terminal) {
        if(real_action != Action::UNDEFINED) {
          const Action bp_translated = translate_pseudo_harmonic(real_action, bp_node->get_branching_actions(), curr_state);
          Logger::log("Blueprint action translation: " + real_action.to_string() + " -> " + bp_translated.to_string());
          _mapped_bp_actions.push_back(bp_translated);
          bp_state = bp_state.apply(bp_translated);
          Logger::log("Blueprint state:\n" + bp_state.to_string());
          curr_state = curr_state.apply(real_action);
          Logger::log("Current state:\n" + curr_state.to_string());
          if(bp_state.get_active() == curr_state.get_active() && bp_state.get_round() == curr_state.get_round()) {
            bp_node = bp_node->apply(bp_translated);
          }
          else {
            Logger::log("Blueprint state mismatch. Forcing terminal solve.");
            force_terminal = true;
          }
        }
        else {
          Logger::log("Live state mismatch. Forcing terminal solve.");
          force_terminal = true;
        }
      }

      const Action live_translated = _mapped_live_actions.get(h_idx);
      oss << pos_to_str(live_state) << " action applied to ranges: " + live_translated.to_string() << ", combos: "
          << std::fixed << std::setprecision(2) << _ranges[live_state.get_active()].n_combos();
      const int expected_cards = n_board_cards(live_state.get_round());
      if(_board.size() < expected_cards) Logger::error("Not enough board cards. Expected="+std::to_string(expected_cards) + ", Board="+cards_to_str(_board));
      update_ranges(_ranges, live_translated, live_state, Board{_board}, decision);
      if(_ranges[live_state.get_active()].n_combos() <= 0) Logger::error("No combos left in " + pos_to_str(live_state) + " range.");
      oss << " -> " << _ranges[live_state.get_active()].n_combos();
      Logger::dump(oss);
      live_state = live_state.apply(live_translated);
      Logger::log("Live state:\n" + live_state.to_string());
    }
    _frozen = std::vector<FrozenNode>{};
  }

  for(int i = 0; i < _ranges.size(); ++i) {
    oss << pos_to_str(i, _ranges.size(), curr_state.is_straddle()) << " card removal, combos: " << _ranges[i].n_combos();
    _ranges[i].remove_cards(_board);
    oss << " -> " << _ranges[i].n_combos();
    Logger::dump(oss);
  }
  _root_state = _real_state;
  _mapped_live_actions = ActionHistory{};
  _live_profile = _init_profiles[_root_state.get_round()];
  Logger::log("New root:\n" + _root_state.to_string());
  if(_root_state.get_action_history().size() != _mapped_bp_actions.size()) {
    Logger::error("Mapped action length mismatch!\nRoot: " + _root_state.get_action_history().to_string() + "\nMapped: " + _mapped_bp_actions.to_string());
  }
  Logger::log("New live profile:\n" + _live_profile.to_string());
  Logger::log("Enqueing solve.");
  if(solve) _enqueue_job(force_terminal);
}

bool Pluribus::_can_solve(const PokerState& root) const {
  return (root.get_round() > 0 || root.active_players() <= 4) && _board.size() >= n_board_cards(root.get_round());
}

void Pluribus::_set_invalid(const std::exception& e) {
  valid = false;
  Logger::log("Exception: " + std::string{e.what()});
  Logger::log("An exception occured while running. Start a new game to attempt to recover.");
}

// bool Pluribus::_should_solve(const PokerState& root) const {
  // return _can_solve(root) && root.get_round() > 0 && (!_solver || !_solver->get_real_time_config().is_terminal());
// }

void Pluribus::_start_worker() {
  if(_solver_thread.joinable()) return;
  _solver_thread = std::thread{[this]{ _solver_worker(); }};
}

void Pluribus::_solver_worker() {
  while(true) {
    std::optional<SolveJob> job;
    std::shared_ptr<TreeRealTimeSolver> local;
    {
      std::unique_lock lk(_solver_mtx);
      _solver_cv.wait(lk, [&]{ return !_running_worker ? true : _pending_job.has_value(); });
      if(!_running_worker) break;
      job.swap(_pending_job);
      job->ack.set_value();
      local = std::make_shared<TreeRealTimeSolver>(job->cfg, job->rt_cfg, _sampled_bp);
      for(const FrozenNode& frozen : _frozen) local->freeze(frozen.freq, frozen.hand, Board{frozen.board}, frozen.live_actions);
      _solver = local;
    }
    local->solve(100'000'000'000L);
  }
}

}
