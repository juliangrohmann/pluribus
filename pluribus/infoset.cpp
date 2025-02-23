#include <cnpy.h>
#include <string>
#include <numeric>
#include <pluribus/util.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/cluster.hpp>
#include <pluribus/infoset.hpp>

namespace pluribus {

std::array<hand_indexer_t, 4> init_indexer_vec() {
  std::array<hand_indexer_t, 4> indexers{4};
  for(int i = 0; i < 4; ++i) init_indexer(indexers[i], i);
  return indexers;
}

const std::array<hand_indexer_t, 4> InformationSet::_indexers = init_indexer_vec();
const std::array<std::vector<uint16_t>, 4> InformationSet::_cluster_map = init_cluster_map(200);

InformationSet::InformationSet(const ActionHistory& history, const Board& board, const Hand& hand, int round) : _history{history} {
  int card_sum = round == 0 ? 2 : 4 + round;
  std::vector<uint8_t> cards(card_sum);
  std::copy(hand.cards().begin(), hand.cards().end(), cards.data());
  if(round > 0) std::copy(board.cards().begin(), board.cards().begin() + card_sum - 2, cards.data() + 2);
  hand_index_t idx = hand_index_last(&_indexers[round], cards.data());
  _cluster = _cluster_map[round][idx];
}

bool InformationSet::operator==(const InformationSet& other) const {
  return _cluster == other._cluster && _history == other._history;
}

std::string InformationSet::to_string() const {
  return "Cluster: " + std::to_string(_cluster) + ", History: " + _history.to_string();
}

long count(const PokerState& state, bool infosets) {
  if(state.is_terminal()) return 0;
  long c;
  if(infosets) {
    c = state.get_round() == 0 ? 169 : 200;
  }
  else {
    c = 1;
  }
  for(Action a : valid_actions(state)) {
    c += count(state.apply(a), infosets);
  }
  return c;
}

long count_infosets(const PokerState& state) {
  return count(state, true);
}

long count_actionsets(const PokerState& state) {
  return count(state, false);
}

}