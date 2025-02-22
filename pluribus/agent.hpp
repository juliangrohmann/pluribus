#pragma once

#include <vector>
#include <random>
#include <pluribus/mccfr.hpp>
#include <pluribus/poker.hpp>

namespace pluribus {

class Agent { 
public:
  virtual Action act(const PokerState& state, const Board& board, const Hand& hand) = 0;
};

class RandomAgent : public Agent {
public:
  Action act(const PokerState& state, const Board& board, const Hand& hand) override;
};

class BlueprintAgent : public Agent {
public:
  BlueprintAgent(const StrategyMap& strategy) : _strategy{strategy} {};
  Action act(const PokerState& state, const Board& board, const Hand& hand) override;
private:
  StrategyMap _strategy; 
};

}
