#pragma once

#include <memory>
#include <pluribus/poker.hpp>
#include <pluribus/blueprint.hpp>

namespace pluribus {
 
enum class SolverState {
  UNDEFINED, SOLVING, SOLVED
};

struct Solution  {
  std::vector<Action> actions;
  std::vector<float> freq;
};

class Solver {
public:
  SolverState get_state() { return _state; }; 
  void solve(const PokerState& state, const std::vector<uint8_t> board, const std::vector<PokerRange>& ranges, 
             const ActionProfile& profile);
  virtual float frequency(const PokerState& state, const Hand& hand, Action action) const = 0;

protected:
  virtual void _solve(const PokerState& state, const std::vector<uint8_t> board, const std::vector<PokerRange>& ranges, 
                      const ActionProfile& profile) = 0;
  
private:
  SolverState _state = SolverState::UNDEFINED;
};

class RealTimeMCCFR : public Solver {
public:
  RealTimeMCCFR(std::shared_ptr<SampledBlueprint> bp) : _bp{bp} {}
  float frequency(const PokerState& state, const Hand& hand, Action action) const override;

protected:
  void _solve(const PokerState& state, const std::vector<uint8_t> board, const std::vector<PokerRange>& ranges, 
                      const ActionProfile& profile) override;

private:
  std::shared_ptr<SampledBlueprint> _bp = nullptr;
  std::unique_ptr<StrategyStorage<int>> _regrets = nullptr;
};

class RealTimeSolver {
public:
  RealTimeSolver(const std::shared_ptr<SampledBlueprint> bp);
  void new_game(int hero_pos);
  void update_state(const PokerState& state);
  void update_board(const std::vector<uint8_t> board);
  Solution solution(const PokerState& state, const Hand& hand);

private:
  void _init_solver();
  void _apply_action(Action a);

  std::shared_ptr<SampledBlueprint> _bp = nullptr;
  std::unique_ptr<Solver> _solver = nullptr;
  PokerState _root_state; // root state has real stack/bet sizes and is not in abstraction 
  PokerState _real_state;
  ActionProfile _live_profile;
  std::vector<PokerRange> _ranges;
  std::vector<uint8_t> _board;
  std::filesystem::path _log_file;
  int _hero_pos = -1;
  int _game_idx = 0;
};
  
}