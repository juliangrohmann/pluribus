#include <pluribus/logging.hpp>
#include <pluribus/pluribus.hpp>
#include <pluribus/translate.hpp>
#include <pluribus/traverse.hpp>

namespace pluribus {

std::string Solution::to_string() const {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(4) << "Solution: actions=" << actions_to_str(actions) << ", freq=[";
  for(int i = 0; i < actions.size(); ++i) oss << freq[i] << (i == freq.size() - 1 ? "]" : ", ");
  return oss.str();
}

std::string FrozenNode::to_string() const {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(4) << "FrozenNode: freq=[";
  for(int i = 0; i < freq.size(); ++i) oss << freq[i] << (i == freq.size() - 1 ? "]" : ", ");
  oss << ", hand=" << hand.to_string() << ", board=" << cards_to_str(board) << ", live_actions=" << live_actions.to_string();
  return oss.str();
}

class RealTimeDecision : public DecisionAlgorithm {
public:
  RealTimeDecision(const LosslessBlueprint& preflop_bp, const std::shared_ptr<const Solver>& solver)
      : _preflop_decision{TreeDecision{preflop_bp.get_strategy(), preflop_bp.get_config().init_state}}, _solver{solver} {}

  float frequency(const Action a, const PokerState& state, const Board& board, const Hand& hand, const int cluster = -1) const override {
    if(_solver) return _solver->frequency(a, state, board, hand);
    if(state.get_round() == 0) return _preflop_decision.frequency(a, state, board, hand);
    Logger::error("Cannot decide postflop frequency without solver.");
  }

private:
  const TreeDecision<float> _preflop_decision;
  const std::shared_ptr<const Solver> _solver;
};

Pluribus::Pluribus(const std::array<ActionProfile, 4>& live_profiles, const std::shared_ptr<const LosslessBlueprint>& preflop_bp,
  const std::shared_ptr<const SampledBlueprint>& sampled_bp)
    : _preflop_bp{preflop_bp}, _sampled_bp{sampled_bp}, _init_profiles{live_profiles} {
  Logger::log("Pluribus action profile:\n" + _sampled_bp->get_config().action_profile.to_string());
  Logger::log((HoleCardIndexer::get_instance() ? "Initialized" : "Failed to initialize") + std::string{" hole card indexer."});
  Logger::log((HandIndexer::get_instance() ? "Initialized" : "Failed to initialize") + std::string{" hand indexer."});
  Logger::log((BlueprintClusterMap::get_instance() ? "Initialized" : "Failed to initialize") + std::string{" blueprint cluster map."});
  // Logger::log((RealTimeClusterMap::get_instance() ? "Initialized" : "Failed to initialize") + std::string{" real time cluster map."});
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

void Pluribus::update_state(const Action action, const int pos) {
  Logger::log("============================== Update State ==============================");
  Logger::log(pos_to_str(pos, _real_state.get_players().size(), _real_state.is_straddle()) + ": " + action.to_string());
  if(_real_state.get_active() != pos) {
    Logger::error("Wrong player is acting. Expected " + pos_to_str(_real_state) + " to act.");
  }
  _apply_action(action, {});
}

void Pluribus::hero_action(const Action action, const std::vector<float>& freq) {
  Logger::log("============================== Hero Action ===============================");
  Logger::log(pos_to_str(_hero_pos, _real_state.get_players().size(), _real_state.is_straddle()) + " (Hero): " + action.to_string());
  if(_real_state.get_active() != _hero_pos) {
    Logger::error("Wrong player is acting. Expected " + pos_to_str(_real_state) + " (hero) to act.");
  }
  _apply_action(action, freq);
}

void Pluribus::update_board(const std::vector<uint8_t>& updated_board) {
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
    _update_root();
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
    Logger::log("Applying blueprint actions to preflop blueprint. Mapped blueprint actions: " + _mapped_bp_actions.to_string());
    const TreeStorageNode<float>* node = _preflop_bp->get_strategy()->apply(_mapped_bp_actions.get_history());
    Logger::log("Applying live actions to preflop blueprint. Mapped live actions: " + _mapped_live_actions.to_string());
    actions = node->apply(_mapped_live_actions.get_history())->get_value_actions();
  }
  Logger::log("Value actions=" + actions_to_str(actions));
  return actions;
}

Solution Pluribus::solution(const Hand& hand) {
  Logger::log("================================ Solution ================================");
  Solution solution;
  const PokerState mapped_state = _root_state.apply(_mapped_live_actions);
  {
    std::lock_guard lk(_solver_mtx);
    solution.actions = _get_solution_actions();
    const RealTimeDecision decision{*_preflop_bp, _solver};
    for(const Action a : solution.actions) {
      solution.freq.push_back(decision.frequency(a, mapped_state, Board{_board}, hand));
    }
  }
  Logger::log(solution.to_string());
  return solution;
}

void Pluribus::save_range(const std::string& fn) {
  Logger::log("=============================== Save Range ===============================");
  PngRangeViewer viewer{fn};
  const PokerState mapped_state = _root_state.apply(_mapped_live_actions);
  {
    std::lock_guard lk(_solver_mtx);
    const RealTimeDecision decision{*_preflop_bp, _solver};
    auto live_ranges = _ranges;
    PokerState curr_state = _root_state;
    for(Action a : _mapped_live_actions.get_history()) {
      std::cout << "Updating range: " << a.to_string() << "\n";
      update_ranges(_ranges, a, curr_state, Board{_board}, decision);
      curr_state.apply_in_place(a);
    }
    std::cout << "Building renderable ranges...\n";
    const std::vector<Action> actions = _get_solution_actions();
    const auto action_ranges = build_renderable_ranges(decision, actions, mapped_state, Board{_board}, live_ranges[mapped_state.get_active()]);
    std::cout << "Rendering ranges...\n";
    render_ranges(&viewer, live_ranges[mapped_state.get_active()], action_ranges);
  }
}

int terminal_round(const PokerState& root) { 
  return root.get_round() >= 2 || (root.get_round() == 1 && root.active_players() == 2) ? 4 : root.get_round() + 1;
}

void Pluribus::_enqueue_job() {
  Logger::log("Initializing solver: MappedRealTimeSolver");
  SolverConfig config{_sampled_bp->get_config().poker, _live_profile};
  config.rake = _sampled_bp->get_config().rake;
  config.init_state = _root_state;
  config.init_board = _board;
  config.init_ranges = _ranges;
  RealTimeSolverConfig rt_config;
  rt_config.bias_profile = BiasActionProfile{};
  rt_config.init_actions = _mapped_bp_actions.get_history();
  rt_config.terminal_round = terminal_round(_root_state);
  rt_config.terminal_bet_level = 999;
  SolveJob job{config, rt_config};
  {
    std::lock_guard lk(_solver_mtx);
    if(_solver) _solver->interrupt();
    _pending_job = std::move(job);
  }
  _solver_cv.notify_one();
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
    oss << std::fixed << std::setprecision(2) << "Max frac difference=" << min_diff << "\nAction size=" << std::setprecision(0) << action_size
        << "\nClosest size=" << closest_size << "\nMax total difference=" << total_diff;
    Logger::dump(oss);
    return true;
  }
  return false;
}

void Pluribus::_apply_action(const Action a, const std::vector<float>& freq) {
  const PokerState prev_real_state = _real_state;
  {
    std::lock_guard lk(_solver_mtx);
    Logger::log("Applying action: " + a.to_string());
    const std::vector<Action> actions = _get_solution_actions();

    _real_state = _real_state.apply(a);
    Logger::log("New state:\n" + _real_state.to_string());

    if(_solver && prev_real_state.get_active() == _hero_pos) {
      const FrozenNode frozen_node{freq, _hero_hand, _board, _mapped_live_actions};
      Logger::log("Freezing hero actions. " + frozen_node.to_string());
      if(freq.size() != actions.size()) {
        Logger::error("Freeze frequency amount mismatch:\nActions=" + actions_to_str(actions));
      }
      _solver->freeze(frozen_node.freq, frozen_node.hand, Board{frozen_node.board}, frozen_node.live_actions);
      _frozen.push_back(frozen_node);
    }
  }

  bool should_solve = false;
  if(_can_solve(_root_state) && is_off_tree(a, actions, prev_real_state)) {
    should_solve = true;
    Logger::log("Action is off-tree. Adding to live actions...");
    _live_profile.add_action(a, prev_real_state.get_round(), prev_real_state.get_bet_level(), prev_real_state.get_active(),
        prev_real_state.is_in_position(prev_real_state.get_active()));
    Logger::log("New live profile:\n" + _live_profile.to_string());
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
  }

  const Action translated = translate_pseudo_harmonic(a, actions, prev_real_state);
  _mapped_live_actions.push_back(translated);
  Logger::log("Live action translation: " + a.to_string() + " -> " + translated.to_string());

  if(_real_state.get_round() > _root_state.get_round() && _can_solve(_real_state)) {
    Logger::log("Round advanced. Updating root...");
    _update_root();
  }
  else if(should_solve) {
    _enqueue_job();
  }
}

void Pluribus::_update_root() {
  std::ostringstream oss;
  PokerState curr_state = _root_state;
  {
    std::lock_guard lk(_solver_mtx);
    const RealTimeDecision decision{*_preflop_bp, _solver};
    const TreeStorageNode<uint8_t>* node = _sampled_bp->get_strategy()->apply(_mapped_bp_actions.get_history());
    for(Action a : _real_state.get_action_history().slice(_root_state.get_action_history().size()).get_history()) {
      const Action translated = translate_pseudo_harmonic(a, node->get_branching_actions(), curr_state);
      _mapped_bp_actions.push_back(translated);
      Logger::log("Blueprint action translation: " + a.to_string() + " -> " + translated.to_string());
      oss << pos_to_str(curr_state) << " action applied to ranges: " + translated.to_string() << ", combos: "
          << std::fixed << std::setprecision(2) << _ranges[curr_state.get_active()].n_combos();
      const int expected_cards = n_board_cards(curr_state.get_round());
      if(_board.size() < expected_cards) Logger::error("Not enough board cards. Expected: " + std::to_string(expected_cards) + ", Board=" + cards_to_str(_board));
      update_ranges(_ranges, a, curr_state, Board{_board}, decision);
      oss << " -> " << _ranges[curr_state.get_active()].n_combos();
      Logger::dump(oss);
      curr_state = curr_state.apply(a);
      node = node->apply(translated);
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
  _enqueue_job();
}

bool Pluribus::_can_solve(const PokerState& root) const {
  return (root.get_round() > 0 || root.active_players() <= 4) && _board.size() >= n_board_cards(root.get_round());
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
    {
      std::unique_lock lk(_solver_mtx);
      _solver_cv.wait(lk, [&]{ return !_running_worker ? true : _pending_job.has_value(); });
      if(!_running_worker) break;
      job.swap(_pending_job);
    }
    const auto local = std::make_shared<TreeRealTimeSolver>(job->cfg, job->rt_cfg, _sampled_bp);
    {
      std::lock_guard lk(_solver_mtx);
      for(FrozenNode frozen : _frozen) local->freeze(frozen.freq, frozen.hand, Board{frozen.board}, frozen.live_actions);
      _solver = local;
    }
    local->solve(100'000'000'000L);
  }
}

}
