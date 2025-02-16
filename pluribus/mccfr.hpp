#pragma once

#include <unordered_map>
#include <pluribus/poker.hpp>
#include <pluribus/infoset.hpp>

namespace pluribus {

class StrategyState {
public:
  StrategyState() : _regret{0.0}, _frequency{0.0} {};
private:
  float _regret;
  float _frequency;
};

class BlueprintTrainer {
public:
  BlueprintTrainer(int strategy_interval, int prune_thresh, int lcfr_thresh, int discount_interval);
  void mccfr_p(int T);
private:
  std::unordered_map<InformationSet, std::unordered_map<uint8_t, StrategyState>> _strategy;
  int _strategy_interval;
  int _prune_thresh;
  int _lcfr_thresh;
  int _discount_interval;
};

}
