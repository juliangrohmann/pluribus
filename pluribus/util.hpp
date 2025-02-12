#include <string>
#include <hand_isomorphism/hand_index.h>

namespace pluribus {

int card_to_idx(const std::string& card);
std::string idx_to_card(int idx);
void str_to_cards(std::string card_str, uint8_t cards[]);
int init_indexer(hand_indexer_t& indexer, int round);

}