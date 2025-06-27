#pragma once

#include <memory>
#include <pluribus/poker.hpp>
#include <pluribus/decision.hpp>
#include <pluribus/blueprint.hpp>

namespace pluribus {
 
enum class SolverState {
  UNDEFINED, INTERRUPT, SOLVING, SOLVED
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
  virtual float frequency(Action action, const PokerState& state, const Board& board, const Hand& hand) const = 0;

protected:
  virtual void _solve(const PokerState& state, const std::vector<uint8_t> board, const std::vector<PokerRange>& ranges, 
                      const ActionProfile& profile) = 0;
  
private:
  SolverState _state = SolverState::UNDEFINED;
};

class RealTimeMCCFR : public Solver {
public:
  RealTimeMCCFR(const std::shared_ptr<const SampledBlueprint> bp) : _bp{bp} {}
  float frequency(Action action, const PokerState& state, const Board& board, const Hand& hand) const override;

protected:
  void _solve(const PokerState& state, const std::vector<uint8_t> board, const std::vector<PokerRange>& ranges, 
                      const ActionProfile& profile) override;

private:
  const std::shared_ptr<const SampledBlueprint> _bp = nullptr;
  std::unique_ptr<StrategyStorage<int>> _regrets = nullptr;
};

class RealTimeDecision : public DecisionAlgorithm {
public:
  RealTimeDecision(const LosslessBlueprint& preflop_bp, const std::shared_ptr<const Solver> solver)
      : _preflop_decision{StrategyDecision{preflop_bp.get_strategy(), preflop_bp.get_config().action_profile}}, _solver{solver} {}

  float frequency(Action a, const PokerState& state, const Board& board, const Hand& hand) const override;

private:
  const StrategyDecision<float> _preflop_decision;
  const std::shared_ptr<const Solver> _solver;
};

class RealTimeSolver {
public:
  RealTimeSolver(const std::shared_ptr<const LosslessBlueprint> preflop_bp, const std::shared_ptr<const SampledBlueprint> sampled_bp);
  void new_game(int hero_pos);
  void update_state(const PokerState& state);
  void update_board(const std::vector<uint8_t> board);
  Solution solution(const PokerState& state, const Hand& hand);

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