#pragma once

#include <memory>
#include <pluribus/poker.hpp>
#include <pluribus/mccfr.hpp>
#include <pluribus/blueprint.hpp>

namespace pluribus {
 
struct Solution  {
  std::vector<Action> actions;
  std::vector<float> freq;
};

class Pluribus {
public:
  Pluribus(const std::shared_ptr<const LosslessBlueprint>& preflop_bp, const std::shared_ptr<const SampledBlueprint>& sampled_bp);
  void new_game(const std::vector<std::string>& players, const std::vector<int>& stacks);
  void update_state(Action action, int pos);
  void update_board(const std::vector<uint8_t> &updated_board);
  Solution solution(const Hand& hand) const;

private:
  void _init_solver();
  void _apply_action(Action a);
  void _update_root();

  const std::shared_ptr<const LosslessBlueprint> _preflop_bp = nullptr;
  const std::shared_ptr<const SampledBlueprint> _sampled_bp = nullptr;
  std::shared_ptr<Solver> _solver = nullptr;
  PokerState _root_state;
  PokerState _real_state;
  ActionHistory _mapped_actions;
  ActionProfile _live_profile;
  std::vector<PokerRange> _ranges;
  std::vector<uint8_t> _board;
  std::filesystem::path _log_file;
  int _hero_pos = -1;
  int _game_idx = 0;
};
  
}