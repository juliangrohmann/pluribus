#pragma once 

#include <pluribus/poker.hpp>
#include <pluribus/rng.hpp>
#include <pluribus/range.hpp>

namespace pluribus {

class HandSampler {
public:
  HandSampler(const std::vector<Hand>& hands, const std::vector<double>& weights);
  HandSampler(const PokerRange& range, bool sparse = false);

  Hand sample() const;

private:
  std::vector<double> _weights_of_range(const PokerRange& range, bool sparse);

  GSLDiscreteDist _dist;
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
