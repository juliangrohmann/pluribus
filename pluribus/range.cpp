#include <pluribus/range.hpp>
#include <pluribus/poker.hpp>

namespace pluribus {

HoleCardIndexer::HoleCardIndexer() {
  int idx = 0;
  for(uint8_t c1 = 0; c1 < 52; ++c1) {
    for(uint8_t c2 = 0; c2 < c1; ++c2) {
      Hand hand{c1, c2};
      _hand_to_idx[hand] = idx;
      _idx_to_hand[idx] = hand;
      ++idx;
    } 
  }
  std::cout << "Indexed " << _hand_to_idx.size() << " hole cards.\n";
}

PokerRange& PokerRange::operator+=(const PokerRange& other) { 
  for(const auto& entry : other.range()) add_hand(entry.first, entry.second); 
  return *this; 
}

PokerRange& PokerRange::operator*=(const PokerRange& other) { 
  for(const auto& entry : other.range()) multiply_hand(entry.first, entry.second); 
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

PokerRange PokerRange::full() {
  PokerRange ret;
  for(uint8_t c1 = 0; c1 < 52; ++c1) {
    for(uint8_t c2 = 0; c2 < c1; ++c2) {
      Hand hand{c1, c2};
      ret.add_hand(hand);
    }
  }
  return ret;
}

}

