#pragma once

#include <memory>
#include <condition_variable>
#include <pluribus/blueprint.hpp>
#include <pluribus/mccfr.hpp>
#include <pluribus/poker.hpp>

namespace pluribus {
 
struct Solution  {
  std::vector<Action> actions;
  std::vector<float> freq;
};

struct FrozenNode {
  std::vector<float> freq;
  Hand hand;
  Board board;
  ActionHistory live_actions;
};

class Pluribus {
public:
  Pluribus(const std::array<ActionProfile, 4>& live_profiles, const std::shared_ptr<const LosslessBlueprint>& preflop_bp,
    const std::shared_ptr<const SampledBlueprint>& sampled_bp);
  ~Pluribus();
  void new_game(const std::vector<int>& stacks, const Hand& hero_hand, int hero_pos);
  void update_state(Action action, int pos);
  void hero_action(Action action, const std::vector<float>& freq);
  void update_board(const std::vector<uint8_t>& updated_board);
  Solution solution(const Hand& hand);
  void save_range(const std::string& fn);

private:
  void _enqueue_job();
  void _apply_action(Action a, const std::vector<float>& freq);
  void _update_root();
  bool _can_solve(const PokerState& root) const;
  // bool _should_solve(const PokerState& root) const;

  void _solver_worker();
  void _start_worker();

  const std::shared_ptr<const LosslessBlueprint> _preflop_bp = nullptr;
  const std::shared_ptr<const SampledBlueprint> _sampled_bp = nullptr;
  std::shared_ptr<TreeRealTimeSolver> _solver = nullptr;
  PokerState _root_state;
  PokerState _real_state;
  ActionHistory _mapped_bp_actions;
  ActionHistory _mapped_live_actions;
  std::array<ActionProfile, 4> _init_profiles;
  ActionProfile _live_profile;
  std::vector<PokerRange> _ranges;
  std::vector<uint8_t> _board;
  std::vector<FrozenNode> _frozen;
  std::filesystem::path _log_file;
  Hand _hero_hand;
  int _hero_pos = -1;
  int _game_idx = 0;

  struct SolveJob {
    SolverConfig cfg;
    RealTimeSolverConfig rt_cfg;
  };

  std::thread _solver_thread;
  std::mutex _solver_mtx;
  std::condition_variable _solver_cv;
  std::optional<SolveJob> _pending_job;
  bool _running_worker = true;
};
  
}