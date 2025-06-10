#include <cnpy.h>
#include <string>
#include <numeric>
#include <memory>
#include <pluribus/util.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/cluster.hpp>
#include <pluribus/history_index.hpp>
#include <pluribus/indexing.hpp>

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

uint64_t HandIndexer::index(const Board& board, const Hand& hand, int round) {
  return index(collect_cards(board, hand, round).data(), round);
}

CachedIndexer::CachedIndexer(int max_round) : _max_round{max_round} {
  hand_indexer_state_init(HandIndexer::get_instance()->get_indexer(max_round), &_state);
}

uint64_t CachedIndexer::index(const uint8_t cards[], int round) {
  while(_indices.size() <= round) {
    int curr_round = _indices.size();
    int offset = curr_round == 0 ? 0 : n_board_cards(curr_round - 1) + 2;
    _indices.push_back(hand_index_next_round(HandIndexer::get_instance()->get_indexer(_max_round), cards + offset, &_state));
  }
  return _indices[round];
}

uint64_t CachedIndexer::index(const Board& board, const Hand& hand, int round) {
  return index(collect_cards(board, hand, round).data(), round);
}

long count(const PokerState& state, const ActionProfile& action_profile, int max_round, bool infosets) {
  if(state.is_terminal() || state.get_round() > max_round) {
    return 0;
  }
  long c;
  if(infosets) {
    c = state.get_round() == 0 ? 169 : 200;
  }
  else {
    c = 1;
  }
  for(Action a : valid_actions(state, action_profile)) {
    c += count(state.apply(a), action_profile, max_round, infosets);
  }
  return c;
}

long count_infosets(const PokerState& state, const ActionProfile& action_profile, int max_round) {
  return count(state, action_profile, max_round, true);
}

long count_actionsets(const PokerState& state, const ActionProfile& action_profile, int max_round) {
  return count(state, action_profile, max_round, false);
}

}