#include <cnpy.h>
#include <string>
#include <numeric>
#include <memory>
#include <pluribus/util.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/cluster.hpp>
#include <pluribus/infoset.hpp>

namespace pluribus {

std::array<hand_indexer_t, 4> init_indexer_vec() {
  std::array<hand_indexer_t, 4> indexers;
  for(int i = 0; i < 4; ++i) init_indexer(indexers[i], i);
  return indexers;
}

std::unique_ptr<HandIndexer> HandIndexer::_instance = nullptr;

HandIndexer::HandIndexer() {
  _indexers = init_indexer_vec();
}

InformationSet::InformationSet(const ActionHistory& history, const Board& board, const Hand& hand, int round, int n_players, int n_chips, int ante) : 
    _history_idx{HistoryIndexer::get_instance()->index(history, n_players, n_chips, ante)} {
  int card_sum = round == 0 ? 2 : 4 + round;
  std::vector<uint8_t> cards(card_sum);
  std::copy(hand.cards().begin(), hand.cards().end(), cards.data());
  if(round > 0) std::copy(board.cards().begin(), board.cards().begin() + card_sum - 2, cards.data() + 2);
  uint64_t idx = HandIndexer::get_instance()->index(cards.data(), round);
  _cluster = FlatClusterMap::get_instance()->cluster(round, idx);
}

InformationSet::InformationSet(const ActionHistory& history, uint16_t cluster, int n_players, int n_chips, int ante) : 
    _history_idx{HistoryIndexer::get_instance()->index(history, n_players, n_chips, ante)}, _cluster{cluster} {}

bool InformationSet::operator==(const InformationSet& other) const {
  return _cluster == other._cluster && _history_idx == other._history_idx;
}

std::string InformationSet::to_string() const {
  return "Cluster: " + std::to_string(_cluster) + ", History index: " + std::to_string(_history_idx);
}

long count(const PokerState& state, int max_round, bool infosets) {
  if(state.is_terminal() || state.get_round() > max_round) return 0;
  long c;
  if(infosets) {
    c = state.get_round() == 0 ? 169 : 200;
  }
  else {
    c = 1;
  }
  for(Action a : valid_actions(state)) {
    c += count(state.apply(a), max_round, infosets);
  }
  return c;
}

long count_infosets(const PokerState& state, int max_round) {
  return count(state, max_round, true);
}

long count_actionsets(const PokerState& state, int max_round) {
  return count(state, max_round, false);
}

}