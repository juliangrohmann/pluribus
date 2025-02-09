#include <string>
#include <cstring>
#include <cmath>
#include <array>
#include <algorithm>
#include <unordered_map>
#include <cassert>
#include <iostream>

#include <pluribus/poker.hpp>

using std::string;
using std::array;
using std::vector;

const string RANKS = "23456789TJQKA";
const string SUITS = "shdc";
const int PRIMES[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41};
const std::vector<int> STRAIGHT_FLUSHES = {7936, 3968, 1984, 992, 496, 248, 124, 62, 31, 4111};
const int MAX_STRAIGHT_FLUSH = 10;
const int MAX_FOUR_OF_A_KIND = 166;
const int MAX_FULL_HOUSE = 322;
const int MAX_FLUSH = 1599;
const int MAX_STRAIGHT = 1609;
const int MAX_THREE_OF_A_KIND = 2467;
const int MAX_TWO_PAIR = 3325;
const int MAX_PAIR = 6185;
const int MAX_HIGH_CARD = 7462;

int prime_prod(const array<int, 5> cards) {
  int product = 1;
  for(int card : cards) {
    product *= card & 0xFF;
  }
  return product;
}

int prime_prod(int rankbits) {
  int product = 1;
  for(int i = 0; i < 13; ++i) {
    if(rankbits & (1 << i)) {
      product *= PRIMES[i];
    }
  }
  return product;
}

void fill_lut(int init_rank, const vector<int>& rankbits, std::unordered_map<int, int>& lut) {
  int rank = init_rank;
  for(int rb : rankbits) {
    int prod = prime_prod(rb);
    lut[prod] = rank++;
  }
}

Evaluator::Evaluator() {
  this->cache_unpaired();
  this->cache_paired();
}

int Evaluator::eval(const array<int, 2>& hand, const array<int, 5>& board) const {
  array<int, 7> cards;
  std::memcpy(cards.data(), hand.data(), sizeof(int) * 2);
  std::memcpy(cards.data() + 2, board.data(), sizeof(int) * 5);

  array<int, 5> sd_hand;
  int minimum = MAX_HIGH_CARD;
  for(int i = 0; i < 7; ++i) {
    for(int j = i + 1; j < 7; ++j) {
      std::memcpy(sd_hand.data(), cards.data(), sizeof(int) * i);
      std::memcpy(sd_hand.data() + i, cards.data() + i + 1, sizeof(int) * (j - i - 1));
      std::memcpy(sd_hand.data() + j - 1, cards.data() + j + 1, sizeof(int) * (6 - j));
      minimum = std::min(minimum, this->eval(sd_hand));
    }
  }
  return minimum;
}

int Evaluator::eval(const array<int, 5>& cards) const {
  // std::cout << "Five cards: " << decode_cards(cards) << std::endl;
  if(cards[0] & cards[1] & cards[2] & cards[3] & cards[4] & 0xF000) {
    int hand_or = (cards[0] | cards[1] | cards[2] | cards[3] | cards[4]) >> 16;
    int prime = prime_prod(hand_or);
    // std::cout << "Flush prime: " << prime << std::endl;
    return this->flush_lut.at(prime);
  }
  else {
    int prime = prime_prod(cards);
    // std::cout << "Unsuited prime: " << prime << std::endl;
    return this->unsuited_lut.at(prime);
  }
  return 1;
}

void Evaluator::cache_unpaired() {
  vector<int> flushes;
  int f = 0b11111;
  for(int i = 0; i < 1277 + STRAIGHT_FLUSHES.size() - 1; ++i) {
    // Bithack: http://www-graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation
    int t = (f | (f - 1)) + 1;
    f = t | ((((t & -t) / (f & -f)) >> 1) - 1);
  
    bool is_str_flush = false;
    for(int sf : STRAIGHT_FLUSHES) {
      is_str_flush |= !(f ^ sf);
    }
    if(!is_str_flush) {
      flushes.push_back(f);
    }
  }

  std::reverse(flushes.begin(), flushes.end());
  fill_lut(1, STRAIGHT_FLUSHES, this->flush_lut); // straight flushes
  fill_lut(MAX_FULL_HOUSE + 1, flushes, this->flush_lut); // flushes
  fill_lut(MAX_FLUSH + 1, STRAIGHT_FLUSHES, this->unsuited_lut); // straights
  fill_lut(MAX_PAIR + 1, flushes, this->unsuited_lut); // high cards
}

void Evaluator::cache_paired() {
  // quads
  int rank = MAX_STRAIGHT_FLUSH + 1;
  for(int quad = 12; quad >= 0; --quad) {
    for(int kicker = 12; kicker >= 0; --kicker) {
      if(quad == kicker) continue;
      int prod = std::pow(PRIMES[quad], 4) * PRIMES[kicker];
      this->unsuited_lut[prod] = rank++;
    }
  }

  // full house
  rank = MAX_FOUR_OF_A_KIND + 1;
  for(int trips = 12; trips >= 0; --trips) {
    for(int pair = 12; pair >= 0; --pair) {
      if(trips == pair) continue;
      int prod = std::pow(PRIMES[trips], 3) * std::pow(PRIMES[pair], 2);
      this->unsuited_lut[prod] = rank++;
    }
  }

  // trips
  rank = MAX_STRAIGHT + 1;
  for(int trips = 12; trips >= 0; --trips) {
    for(int high_kicker = 12; high_kicker >= 0; --high_kicker) {
      for(int low_kicker = high_kicker - 1; low_kicker >= 0; --low_kicker) {
        if(trips == high_kicker || trips == low_kicker) continue;
        int prod = std::pow(PRIMES[trips], 3) * PRIMES[high_kicker] * PRIMES[low_kicker];
        this->unsuited_lut[prod] = rank++;
      }
    }
  }

  // two pair
  rank = MAX_THREE_OF_A_KIND + 1;
  for(int high_pair = 12; high_pair >= 0; --high_pair) {
    for(int low_pair = high_pair - 1; low_pair >= 0; --low_pair) {
      for(int kicker = 12; kicker >= 0; --kicker) {
        if(high_pair == kicker || low_pair == kicker) continue;
        int prod = std::pow(PRIMES[high_pair], 2) * std::pow(PRIMES[low_pair], 2) * PRIMES[kicker];
        this->unsuited_lut[prod] = rank++;
      }
    }
  }

  // pair
  rank = MAX_TWO_PAIR + 1;
  for(int pair = 12; pair >= 0; --pair) {
    for(int top_kicker = 12; top_kicker >= 0; --top_kicker) {
      for(int mid_kicker = top_kicker - 1; mid_kicker >= 0; --mid_kicker) {
        for(int bot_kicker = mid_kicker - 1; bot_kicker >= 0; --bot_kicker) {
          if(pair == top_kicker || pair == mid_kicker || pair == bot_kicker) continue;
          int prod = std::pow(PRIMES[pair], 2) * PRIMES[top_kicker] * PRIMES[mid_kicker] * PRIMES[bot_kicker];
          this->unsuited_lut[prod] = rank++;
        }
      }
    }
  }
}

int make_card(const string& card) {
  int rank_idx = RANKS.find(card[0]);
  int suit_idx = SUITS.find(card[1]);
  int rank_prime = PRIMES[rank_idx];
  int bitrank = 1 << rank_idx << 16;
  int suit = 1 << suit_idx << 12;
  int rank = rank_idx << 8;
  return bitrank | suit | rank | rank_prime;
}

array<int, 2> make_hand(const string& str) {
  assert(str.length() == 4 && "Hand string must be 4 characters.");
  return {make_card(str.substr(0, 2)), make_card(str.substr(2, 2))};
}

array<int, 5> make_board(const string& str) {
  assert(str.length() % 2 == 0 && "Board string must have an even amount of characters.");
  assert(str.length() <= 10 && "Board string must be at most 10 characters.");
  array<int, 5> board = {-1};
  for(int i = 0; i < str.length(); i += 2) {
    board[i / 2] = make_card(str.substr(i, 2));
  }
  return board;
}

string decode_cards(int bits) {
  int rank_idx = (bits >> 8) & 0xF;
  int suit_idx = std::log2((bits >> 12) & 0xF);
  return string(1, RANKS[rank_idx]) + SUITS[suit_idx];
}
