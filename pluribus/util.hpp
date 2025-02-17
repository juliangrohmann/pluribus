#pragma once

#include <string>
#include <hand_isomorphism/hand_index.h>

namespace pluribus {

uint8_t card_to_idx(const std::string& card);
std::string idx_to_card(int idx);
void str_to_cards(std::string card_str, uint8_t cards[]);
std::string cards_to_str(const uint8_t cards[], int n);
int init_indexer(hand_indexer_t& indexer, int round);

}