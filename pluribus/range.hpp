#pragma once

#include <unordered_map>
#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>
#include <pluribus/poker.hpp>

namespace pluribus {

class HoleCardIndexer {
public:
  uint16_t index(const Hand& hand) const { 
    return _hand_to_idx.at(canonicalize(hand)); 
  }
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
  PokerRange(double freq = 0.0f) : _weights(1326, freq) { HoleCardIndexer::get_instance(); }

  void add_hand(const Hand& hand, double freq = 1.0f) { 
    int idx = HoleCardIndexer::get_instance()->index(hand);
    if(idx >= _weights.size()) {
      throw std::runtime_error{"PokerRange writing out of bounds!" + std::to_string(idx) + " < " + std::to_string(_weights.size())};
    }
    _weights[idx] += freq; 
  }
  void multiply_hand(const Hand& hand, double freq) { _weights[HoleCardIndexer::get_instance()->index(hand)] *= freq; }
  void set_frequency(const Hand& hand, double freq) { _weights[HoleCardIndexer::get_instance()->index(hand)] = freq; }
  double frequency(const Hand& hand) const { return _weights[HoleCardIndexer::get_instance()->index(hand)]; }
  const std::vector<double>& weights() const { return _weights; }
  void remove_cards(const std::vector<uint8_t>& cards);
  std::vector<Hand> hands() const;
  double n_combos() const;
  Hand sample(std::unordered_set<uint8_t> dead_cards = {}) const;

  PokerRange& operator+=(const PokerRange& other);
  PokerRange& operator*=(const PokerRange& other);
  PokerRange operator+(const PokerRange& other) const;
  PokerRange operator*(const PokerRange& other) const;
  bool operator==(const PokerRange&) const = default;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_weights);
  }

  static PokerRange full() { return PokerRange{1.0f}; }
  static PokerRange random();
private:
  std::vector<double> _weights;
};

}