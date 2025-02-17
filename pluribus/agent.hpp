#pragma once

#include <vector>
#include <random>
#include <pluribus/poker.hpp>

namespace pluribus {

class Agent { 
public:
  virtual Action act(const PokerState& state) = 0;
};

class RandomAgent : public Agent {
public:
  Action act(const PokerState& state) override;
};

}
