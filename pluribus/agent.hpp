#pragma once

#include <utility>
#include <vector>
#include <pluribus/mccfr.hpp>
#include <pluribus/poker.hpp>

namespace pluribus {

class Agent {
public:
  virtual ~Agent() = default;

  virtual Action act(const PokerState& state, const Board& board, const Hand& hand, const PokerConfig& config) = 0;
};

class RandomAgent : public Agent {
public:
  explicit RandomAgent(ActionProfile  action_profile) : _action_profile{std::move(action_profile)} {}
  Action act(const PokerState& state, const Board& board, const Hand& hand, const PokerConfig& config) override;
private:
  ActionProfile _action_profile;
};

class BlueprintAgent : public Agent {
public:
  explicit BlueprintAgent(const MappedBlueprintSolver* trainer_p) : _trainer_p{trainer_p} {}
  Action act(const PokerState& state, const Board& board, const Hand& hand, const PokerConfig& config) override;
private:
  const MappedBlueprintSolver* _trainer_p;
};

void evaluate_agents(const std::vector<Agent*>& agents, const PokerConfig& config, long n_iter);
void evaluate_strategies(const std::vector<MappedBlueprintSolver*>& strategies, long n_iter);
void evaluate_vs_random(const MappedBlueprintSolver* hero, long n_iter);

}
