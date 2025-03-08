#include <numeric>
#include <pluribus/rng.hpp>
#include <pluribus/range.hpp>
#include <pluribus/poker.hpp>

namespace pluribus {

HoleCardIndexer::HoleCardIndexer() {
  int idx = 0;
  for(uint8_t c1 = 0; c1 < 52; ++c1) {
    for(uint8_t c2 = 0; c2 < c1; ++c2) {
      Hand hand = canonicalize(Hand{c1, c2});
      // std::cout << "Initialized: " << hand.to_string() << "\n";
      _hand_to_idx[hand] = idx;
      _idx_to_hand[idx] = hand;
      if(_hand_to_idx.find(hand) == _hand_to_idx.end()) {
        std::cout << "failed init: " << hand.to_string() << "\n";
      }
      // std::cout << "Missing hand=" << (_hand_to_idx.find(hand) != _hand_to_idx.end()) << ", Missing idx=" << (_idx_to_hand.find(idx) != _idx_to_hand.end()) << "\n";
      ++idx;
    } 
  }
  std::cout << "Indexed " << _hand_to_idx.size() << " hole cards.\n";
}

std::unique_ptr<HoleCardIndexer> HoleCardIndexer::_instance = nullptr;

float PokerRange::n_combos() const {
  return std::reduce(_weights.begin(), _weights.end());
}

Hand PokerRange::sample(std::unordered_set<uint8_t> dead_cards) const {
  // TODO: Alias Method with Precomputed Table for performance
  std::vector<float> masked_weights = _weights;
  for(int i = 0; i < masked_weights.size(); ++i) {
    Hand hand = HoleCardIndexer::get_instance()->hand(i);
    if(dead_cards.find(hand.cards()[0]) != dead_cards.end() || dead_cards.find(hand.cards()[1]) != dead_cards.end()) {
      masked_weights[i] = 0.0f;
    }
  }
  std::discrete_distribution<> dist(masked_weights.begin(), masked_weights.end());
  return HoleCardIndexer::get_instance()->hand(dist(GlobalRNG::instance()));
}

PokerRange& PokerRange::operator+=(const PokerRange& other) { 
  for(int i = 0; i < _weights.size(); ++i) _weights[i] += other._weights[i];
  return *this; 
}

PokerRange& PokerRange::operator*=(const PokerRange& other) { 
  for(int i = 0; i < _weights.size(); ++i) _weights[i] *= other._weights[i];
  return *this; 
}

PokerRange PokerRange::operator+(const PokerRange& other) const {
  PokerRange ret = *this;
  ret += other;
  return ret;
}

PokerRange PokerRange::operator*(const PokerRange& other) const {
  PokerRange ret = *this;
  ret *= other;
  return ret;
}

}

