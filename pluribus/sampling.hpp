#pragma once 

#include "ransampl.h"
#include <pluribus/poker.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/range.hpp>

namespace pluribus {

class HandSampler {
public:
  HandSampler(const std::vector<Hand>& hands, const std::vector<double>& weights);
  HandSampler(const PokerRange& range, bool sparse = false);

  Hand sample() const;
  void remap(const std::vector<double>& weights);
  void remap(const PokerRange& range) { remap(range.weights()); };

private:
  void _init(const std::vector<double>& weights);
  std::vector<double> _sparse_weights(const PokerRange& range);

  std::shared_ptr<ransampl_ws> _state = nullptr;
  std::vector<Hand> _hands;
};

class RoundSampler {
public:
  RoundSampler(const std::vector<PokerRange>& ranges);
  std::vector<Hand> sample(const std::vector<uint8_t>& dead_cards = {}, const std::vector<Player>* players = nullptr);
  long _samples = 0;
  long _rejections = 0;
private:
  std::vector<HandSampler> _samplers;
  
};

}
