#pragma once 

#include "ransampl.h"
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

  ~HandSampler();

private:
  void _init(const std::vector<double>& weights);
  std::vector<double> _sparse_weights(const PokerRange& range);

  ransampl_ws* _state = nullptr;
  std::vector<Hand> _hands;
};

}
