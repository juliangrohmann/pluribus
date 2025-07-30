#include <numeric>
#include <omp/Hand.h>
#include <pluribus/constants.hpp>
#include <pluribus/logging.hpp>
#include <pluribus/mccfr.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/range.hpp>
#include <pluribus/rng.hpp>

namespace pluribus {

HoleCardIndexer::HoleCardIndexer() {
  _idx_to_hand.resize(MAX_COMBOS);
  int idx = 0;
  for(uint8_t c1 = 0; c1 < MAX_CARDS; ++c1) {
    for(uint8_t c2 = 0; c2 < c1; ++c2) {
      Hand hand = canonicalize(Hand{c1, c2});
      // std::cout << "Initialized: " << hand.to_string() << "\n";
      _hand_to_idx[hand] = idx;
      _idx_to_hand[idx] = hand;
      if(!_hand_to_idx.contains(hand)) {
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
  for(uint8_t c1 = 0; c1 < MAX_CARDS; ++c1) {
    for(uint8_t c2 = 0; c2 < c1; ++c2) {
      if(Hand hand = canonicalize(Hand{c1, c2}); frequency(hand) > 0) ret.push_back(hand);
    } 
  }
  return ret;
}

double PokerRange::n_combos() const {
  return std::reduce(_weights.begin(), _weights.end());
}

std::string PokerRange::to_string() const {
  std::ostringstream oss;
  for(int i = 0; i < MAX_COMBOS; ++i) {
    Hand hand = HoleCardIndexer::get_instance()->hand(i);
    oss << hand.to_string() << ": " << frequency(hand) << "\n";
  }
  return oss.str();
}

void PokerRange::normalize() {
  const double sum = n_combos();
  for(auto& w : _weights) w /= sum;
}

void PokerRange::make_relative() {
  double max_w = 0.0;
  for(const auto& w : _weights) max_w = std::max(w, max_w);
  std::cout << std::fixed << std::setprecision(8) << max_w << "\n";
  for(auto& w : _weights) w /= max_w;
}

void PokerRange::remove_cards(const std::vector<uint8_t>& cards) {
  for(const auto& hand : hands()) {
    if(collides(hand, cards)) {
      set_frequency(hand, 0.0);
    }
  }
}

PokerRange PokerRange::bayesian_update(const PokerRange& prior_range, const PokerRange& action_range) const {
  PokerRange updated = *this;
  double p_a = 0.0;
  const PokerRange post_range = prior_range * action_range;
  for(int i = 0; i < updated._weights.size(); ++i) {
    Hand hand = HoleCardIndexer::get_instance()->hand(i);
    auto blocked_post = post_range;
    std::vector<uint8_t> blocked = {hand.cards().begin(), hand.cards().end()};
    blocked_post.remove_cards(blocked);
    const auto a_given_h = blocked_post.n_combos() / prior_range.n_combos();
    p_a += a_given_h * _weights[i];
    // std::cout << "P(fold|" << hand.to_string() << ")=" << a_given_h <<"\n";
    updated._weights[i] *= a_given_h;
  }
  p_a /= n_combos();
  // std::cout << "P(a)=" << p_a << "\n";
  for(auto& w : updated._weights) w /= p_a;
  return updated;
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
  for(uint8_t i = 0; i < MAX_CARDS; ++i) {
    for(uint8_t j = i + 1; j < MAX_CARDS; ++j) {
      range.add_hand(Hand{j, i}, GlobalRNG::uniform());
    }
  }
  return range;
}

std::vector<Hand> select_by_suit(const char primary, const char kicker, const bool suited) {
  std::unordered_set<Hand> hands;
  for(const auto& prim_suit : omp::SUITS) {
    for(const auto& kick_suit : omp::SUITS) {
      if(const bool match = prim_suit == kick_suit; match == suited) {
        hands.insert(canonicalize(Hand{std::string{primary} + prim_suit + kicker + kick_suit}));
      }
    }
  }
  return std::vector<Hand>{hands.begin(), hands.end()};
}

std::vector<Hand> all_suits(const char primary, const char kicker) {
  auto suited_hands = select_by_suit(primary, kicker, true);
  auto offsuit_hands = select_by_suit(primary, kicker, false);
  std::vector<Hand> hands{suited_hands.begin(), suited_hands.end()};
  hands.insert(hands.end(), offsuit_hands.begin(), offsuit_hands.end());
  return hands;
}

void set_hand(PokerRange& range, const std::string& hand, const double freq) {
  if(hand.length() == 4) {
    range.set_frequency(Hand{hand}, freq);
  }
  std::vector<Hand> hands;
  if(hand.length() == 2) {
    hands = all_suits(hand[0], hand[1]);
  }
  else if(hand.length() == 3) {
    if(hand[2] != 's' && hand[2] != 'o') Logger::error("Invalid hand suit specifier: " + hand);
    hands = select_by_suit(hand[0], hand[1], hand[2] == 's');
  }
  else {
    Logger::error("Invalid hand: " + hand);
  }
  for(const auto& h : hands) {
    range.set_frequency(h, freq);
  }
}

int rank_index(const char rank) {
  return std::ranges::distance(omp::RANKS.begin(), std::ranges::find(omp::RANKS, rank));
}

void set_hand_range(PokerRange& range, const char primary, const char start_kicker, const char end_kicker, const std::string& suit_spec, const double freq) {
  if(!suit_spec.empty() && suit_spec != "s" && suit_spec != "o") Logger::error("invalid suit spec: " + suit_spec);
  const int start_idx = rank_index(start_kicker);
  const int end_idx = rank_index(end_kicker);
  if(start_idx > end_idx) Logger::error("Invalid kicker range: " + primary + start_kicker + std::string{" to "} + primary + end_kicker);
  for(int i = start_idx; i <= end_idx; ++i) {
    set_hand(range, std::string{primary} + omp::RANKS[i] + suit_spec, freq);
  }
}

}

