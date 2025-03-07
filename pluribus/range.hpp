#pragma once

#include <unordered_map>
#include <pluribus/poker.hpp>

namespace pluribus {

class HoleCardIndexer {
public:
  uint16_t index(const Hand& hand) const { return _hand_to_idx.at(canonicalize(hand)); }
  Hand hand(uint16_t idx) const { return _idx_to_hand.at(idx); }

  static HoleCardIndexer* get_instance() {
    if(!_instance) {
      _instance = std::unique_ptr<HoleCardIndexer>(new HoleCardIndexer());
    }
    return _instance.get();
  }

  HoleCardIndexer(const HoleCardIndexer&) = delete;
  HoleCardIndexer& operator==(const HoleCardIndexer&) = delete;

private:
  HoleCardIndexer();

  std::unordered_map<Hand, uint16_t> _hand_to_idx;
  std::unordered_map<uint16_t, Hand> _idx_to_hand;

  static std::unique_ptr<HoleCardIndexer> _instance;
};

class PokerRange {
public:
  void add_hand(const Hand& hand, float freq = 1.0f) { _range[canonicalize(hand)] += freq; }
  float frequency(const Hand& hand) const { return _range.at(canonicalize(hand)); }
  const std::unordered_map<Hand, float>& range() const { return _range; }
private:
  std::unordered_map<Hand, float> _range;
};

}