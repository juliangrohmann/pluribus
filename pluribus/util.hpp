#pragma once

#include <string>
#include <cereal/cereal.hpp>
#include <hand_isomorphism/hand_index.h>

namespace pluribus {

std::string date_time_str();
uint8_t card_to_idx(const std::string& card);
std::string idx_to_card(int idx);
void str_to_cards(std::string card_str, uint8_t cards[]);
std::string cards_to_str(const uint8_t cards[], int n);
int init_indexer(hand_indexer_t& indexer, int round);

template <class T>
void cereal_save(const T& strategy, const std::string& fn) {
  std::cout << "Saving to " << fn << '\n';
  std::ofstream os(fn, std::ios::binary);
  cereal::BinaryOutputArchive oarchive(os);
  oarchive(strategy);
}

template <class T>
T cereal_load(const std::string& fn) {
  std::cout << "Loading from " << fn << '\n';
  std::ifstream is(fn, std::ios::binary);
  cereal::BinaryInputArchive iarchive(is);
  T data;
  iarchive(data);
  return data;
}

}