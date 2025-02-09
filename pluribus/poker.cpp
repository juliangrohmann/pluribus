#include <string>
#include <cassert>
#include <omp/Hand.h>
#include <pluribus/poker.hpp>

namespace pluribus {

int card_to_idx(const std::string& card) {
  assert(card.length() == 2 && "Card string must have length == 2.");
  return omp::RANKS.find(card[0]) * 4 + omp::SUITS.find(card[1]);
}

std::string idx_to_card(int idx) {
  return std::string(1, omp::RANKS[idx / 4]) + omp::SUITS[idx % 4];
}

}
