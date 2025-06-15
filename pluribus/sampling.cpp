#include "ransampl.h"
#include <pluribus/rng.hpp>
#include <pluribus/sampling.hpp>

namespace pluribus {

void HandSampler::_init(const std::vector<double>& weights) {
  _state = std::shared_ptr<ransampl_ws>{ransampl_alloc(weights.size(), &GSLGlobalRNG::uniform), [](ransampl_ws* s) { ransampl_free(s); }};
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
  ransampl_set(_state.get(), weights.data());
}

Hand HandSampler::sample() const {
  int idx = ransampl_draw(_state.get());
  return _hands.size() == 0 ? HoleCardIndexer::get_instance()->hand(idx) : _hands[idx];
}

RoundSampler::RoundSampler(const std::vector<PokerRange>& ranges) {
 _samplers.reserve(ranges.size());
 for(const auto& r : ranges) _samplers.emplace_back(r);
}

bool _is_valid(const std::vector<Hand>& hands, const std::vector<uint8_t>& dead_cards, const std::vector<Player>* players) {
  std::unordered_set<uint8_t> cards{dead_cards.begin(), dead_cards.end()};
  int added = cards.size();
  for(int i = 0; i < hands.size(); ++i) {
    if(!players || !players->operator[](i).has_folded()) {
      cards.insert(hands[i].cards()[0]);
      cards.insert(hands[i].cards()[1]);
      added += 2;
      if(cards.size() != added) return false;
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