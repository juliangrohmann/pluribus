#include "ransampl.h"
#include <pluribus/rng.hpp>
#include <pluribus/sampling.hpp>

namespace pluribus {

void HandSampler::_init(const std::vector<double>& weights) {
  _state = ransampl_alloc(weights.size(), &GlobalRNG::uniform);
  remap(weights);
}

std::vector<double> HandSampler::_sparse_weights(const PokerRange& range) {
  std::vector<double> weights;
  for(uint8_t c1 = 0; c1 < 52; ++c1) {
    for(uint8_t c2 = 0; c2 < c1; ++c2) {
      Hand hand = canonicalize(Hand{c1, c2});
      double freq = range.frequency(hand);
      if(freq > 0) {
        _hands.push_back(hand);
        weights.push_back(freq);
      };
    } 
  }
  return weights;
}

HandSampler::HandSampler(const std::vector<Hand>& hands, const std::vector<double>& weights) : _hands{hands} {
  _init(weights);
}

HandSampler::HandSampler(const PokerRange& range, bool sparse) {
  if(sparse) {
    _init(_sparse_weights(range));
  }
  else {
    _init(range.weights());
  }
}

void HandSampler::remap(const std::vector<double>& weights) {
  if(weights.size() != _state->n) throw std::runtime_error("Cannot change size of Hand distribution during remap.");
  ransampl_set(_state, weights.data());
}

Hand HandSampler::sample() const {
  int idx = ransampl_draw(_state);
  return _hands.size() == 0 ? HoleCardIndexer::get_instance()->hand(idx) : _hands[idx];
}

HandSampler::~HandSampler() {
  ransampl_free(_state);
}

}