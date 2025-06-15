#include "ransampl.h"
#include <pluribus/rng.hpp>
#include <pluribus/sampling.hpp>

namespace pluribus {

std::vector<double> HandSampler::_weights_of_range(const PokerRange& range, bool sparse) {
  if(!sparse) return range.weights();
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

HandSampler::HandSampler(const std::vector<Hand>& hands, const std::vector<double>& weights) : _hands{hands}, _dist{weights} {}

HandSampler::HandSampler(const PokerRange& range, bool sparse) : _dist{_weights_of_range(range, sparse)} {}

Hand HandSampler::sample() const {
  int idx = _dist.sample();
  return _hands.size() == 0 ? HoleCardIndexer::get_instance()->hand(idx) : _hands[idx];
}

RoundSampler::RoundSampler(const std::vector<PokerRange>& ranges) {
 _samplers.reserve(ranges.size());
 for(const auto& r : ranges) _samplers.emplace_back(r);
}

bool _is_valid(const std::vector<Hand>& hands, const std::vector<uint8_t>& dead_cards, const std::vector<Player>* players) {
  uint64_t mask = 0;
  for(uint8_t card : dead_cards) mask |= 1L << card;
  for(int i = 0; i < hands.size(); ++i) {
    if(!players || !players->operator[](i).has_folded()) {
      uint64_t curr_mask = (1L << hands[i].cards()[0]) | (1L << hands[i].cards()[1]);
      if(mask & curr_mask) return false;
      mask |= curr_mask;
    }
  }
  return true;
}

std::vector<Hand> RoundSampler::sample(const std::vector<uint8_t>& dead_cards, const std::vector<Player>* players) {
  std::vector<Hand> hands(_samplers.size(), Hand{52, 52});
  bool rejected = false;
  do {
    for(int i = 0; i < hands.size(); ++i) {
      if(!players || !players->operator[](i).has_folded()) {
        hands[i] = _samplers[i].sample();
      }
    }
    if(rejected) ++_rejections;
    ++_samples;
    rejected = true;
  } while(!_is_valid(hands, dead_cards, players));
  return hands;
}

}