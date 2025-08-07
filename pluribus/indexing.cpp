#include <cnpy.h>
#include <memory>
#include <numeric>
#include <string>
#include <pluribus/cluster.hpp>
#include <pluribus/indexing.hpp>
#include <pluribus/logging.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/util.hpp>

namespace pluribus {

std::unique_ptr<HandIndexer> HandIndexer::_instance = nullptr;

HandIndexer::HandIndexer() {
  for(int i = 0; i < 4; ++i) init_indexer(_indexers[i], i);
}

uint64_t HandIndexer::index(const Board& board, const Hand& hand, const int round) const {
  return index(collect_cards(board, hand, round).data(), round);
}

std::unique_ptr<FlopIndexer> FlopIndexer::_instance = nullptr;

FlopIndexer::FlopIndexer() {
  constexpr uint8_t cards_per_round[1] = {3};
  hand_indexer_init(1, cards_per_round, &_indexer);
  const hand_index_t num_flops = hand_indexer_size(&_indexer, 0);
  if(num_flops != NUM_DISTINCT_FLOPS) {
    Logger::error("Flop indexer size mismatch: Current=" + std::to_string(num_flops) + ", Expected=" + std::to_string(NUM_DISTINCT_FLOPS));
  }
}

CachedIndexer::CachedIndexer(const int max_round) : _max_round{max_round} {
  hand_indexer_state_init(HandIndexer::get_instance()->get_indexer(max_round), &_state);
}

uint64_t CachedIndexer::index(const uint8_t cards[], const int round) {
  while(static_cast<int>(_indices.size()) <= round) {
    const int curr_round = static_cast<int>(_indices.size());
    const int offset = curr_round == 0 ? 0 : n_board_cards(curr_round - 1) + 2;
    _indices.push_back(hand_index_next_round(HandIndexer::get_instance()->get_indexer(_max_round), cards + offset, &_state));
  }
  return _indices[round];
}

uint64_t CachedIndexer::index(const Board& board, const Hand& hand, const int round) {
  return _indices.size() > round ? _indices[round] : index(collect_cards(board, hand, round).data(), round);
}

long count(const PokerState& state, const ActionProfile& action_profile, const int max_round, const bool infosets) {
  if(state.is_terminal() || state.get_round() > max_round) {
    return 0;
  }
  long c;
  if(!infosets) {
    c = state.get_round() == 0 ? 169 : 200;
  }
  else {
    c = 1;
  }
  for(const Action a : valid_actions(state, action_profile)) {
    c += count(state.apply(a), action_profile, max_round, infosets);
  }
  return c;
}

long count_infosets(const PokerState& state, const ActionProfile& action_profile, const int max_round) {
  return count(state, action_profile, max_round, true);
}

long count_actionsets(const PokerState& state, const ActionProfile& action_profile, const int max_round) {
  return count(state, action_profile, max_round, false);
}

}