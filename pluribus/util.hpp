#pragma once

#include <string>
#include <fstream>
#include <cereal/cereal.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <cereal/archives/binary.hpp>
#include <hand_isomorphism/hand_index.h>

namespace pluribus {

long long get_free_ram();
std::string date_time_str();
uint8_t card_to_idx(const std::string& card);
std::string idx_to_card(int idx);
void str_to_cards(std::string card_str, uint8_t cards[]);
std::string cards_to_str(const uint8_t cards[], int n);
int init_indexer(hand_indexer_t& indexer, int round);

}