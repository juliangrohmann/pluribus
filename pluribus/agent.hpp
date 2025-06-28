#pragma once

#include <vector>
#include <random>
#include <pluribus/mccfr.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/storage.hpp>

namespace pluribus {

class Agent { 
public:
  virtual Action act(const PokerState& state, const Board& board, const Hand& hand, const PokerConfig& config) = 0;
};

class RandomAgent : public Agent {
public:
  RandomAgent(const ActionProfile& action_profile) : _action_profile{action_profile} {};
  Action act(const PokerState& state, const Board& board, const Hand& hand, const PokerConfig& config) override;
private:
  ActionProfile _action_profile;
};

class BlueprintAgent : public Agent {
public:
  BlueprintAgent(const BlueprintSolver* trainer_p) : _trainer_p{trainer_p} {};
  Action act(const PokerState& state, const Board& board, const Hand& hand, const PokerConfig& config) override;
private:
  const BlueprintSolver* _trainer_p;
};

void evaluate_agents(const std::vector<Agent*>& agents, const PokerConfig& config, long n_iter);
void evaluate_strategies(const std::vector<BlueprintSolver*>& strategies, long n_iter);
void evaluate_vs_random(const BlueprintSolver* hero, long n_iter);

// class SampledBlueprintAgent : public Agent {
// public:
//   SampledBlueprintAgent(const BlueprintSolver& trainer);
//   Action act(const PokerState& state, const Board& board, const Hand& hand, const PokerConfig& config) override;
// private:
//   void populate(const PokerState& state, const BlueprintSolver& trainer);

//   ActionStorage _strategy;
//   ActionProfile _action_profile;
// };

}
