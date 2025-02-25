#pragma once

#include <vector>
#include <random>
#include <pluribus/mccfr.hpp>
#include <pluribus/poker.hpp>

namespace pluribus {

class Agent { 
public:
  virtual Action act(const PokerState& state, const Board& board, const Hand& hand, int n_players, int n_chips, int ante) = 0;
};

class RandomAgent : public Agent {
public:
  Action act(const PokerState& state, const Board& board, const Hand& hand, int n_players, int n_chips, int ante) override;
};

// class BlueprintAgent : public Agent {
// public:
//   BlueprintAgent(BlueprintTrainer& trainer);
//   Action act(const PokerState& state, const Board& board, const Hand& hand, int n_players, int n_chips, int ante) override;
// private:
//   std::unordered_map<InformationSet, Action> _strategy;
//   void populate(const PokerState& state, BlueprintTrainer& trainer);
// };

}
