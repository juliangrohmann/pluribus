#include <numeric>
#include <pluribus/rng.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/mccfr.hpp>
#include <pluribus/range.hpp>

namespace pluribus {

HoleCardIndexer::HoleCardIndexer() {
  _idx_to_hand.resize(1326);
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

std::vector<Hand> PokerRange::hands() const {
  std::vector<Hand> ret;
  for(uint8_t c1 = 0; c1 < 52; ++c1) {
    for(uint8_t c2 = 0; c2 < c1; ++c2) {
      Hand hand = canonicalize(Hand{c1, c2});
      if(frequency(hand) > 0) ret.push_back(hand);
    } 
  }
  return ret;
}

double PokerRange::n_combos() const {
  return std::reduce(_weights.begin(), _weights.end());
}

void PokerRange::remove_cards(const std::vector<uint8_t>& cards) {
  for(const auto& hand : hands()) {
    if(collides(hand, cards)) {
      set_frequency(hand, 0.0);
    }
  }
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

PokerRange PokerRange::random() {
  PokerRange range{0.0};
  for(uint8_t i = 0; i < 52; ++i) {
    for(uint8_t j = i + 1; j < 52; ++j) {
      range.add_hand(Hand{j, i}, GlobalRNG::uniform());
    }
  }
  return range;
}

}

