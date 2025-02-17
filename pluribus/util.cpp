#include <iostream>
#include <iomanip>
#include <cassert>
#include <string>
#include <ctime>
#include <sstream>
#include <omp/Hand.h>
#include <hand_isomorphism/hand_index.h>

namespace pluribus {

std::string date_time_str() {
    std::time_t t = std::time(nullptr);
    std::tm tm;
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
    return oss.str();
}

uint8_t card_to_idx(const std::string& card) {
  assert(card.length() == 2 && "Card string must have length == 2.");
  return omp::RANKS.find(card[0]) * 4 + omp::SUITS.find(card[1]);
}

std::string idx_to_card(int idx) {
  return std::string(1, omp::RANKS[idx / 4]) + omp::SUITS[idx % 4];
}

void str_to_cards(std::string card_str, uint8_t cards[]) {
  for(int i = 0; i < card_str.length(); i += 2) {
    cards[i / 2] = card_to_idx(card_str.substr(i, 2));
  }
}

std::string cards_to_str(const uint8_t cards[], int n) {
  std::string str = "";
  for(int i = 0; i < n; ++i) {
    str += idx_to_card(cards[i]);
  }
  return str;
}

int init_indexer(hand_indexer_t& indexer, int round) {
  uint8_t n_cards[round + 1];
  uint8_t all_rounds[] = {2, 3, 1, 1};
  int card_sum = 0;
  for(int i = 0; i < round + 1; ++i) {
    n_cards[i] = all_rounds[i];
    card_sum += all_rounds[i];
  }
  bool init_success = hand_indexer_init(round + 1, n_cards, &indexer);
  assert(init_success && "Failed to initialize indexer.");
  return card_sum;
}

}