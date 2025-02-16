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
  RandomAgent() : _rng{std::random_device{}()} {}
  // RandomAgent() : _rng{222} {}
  Action act(const PokerState& state) override;

private:
  std::mt19937 _rng;
};

}
