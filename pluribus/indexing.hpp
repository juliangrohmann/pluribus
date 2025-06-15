#pragma once

#include <hand_isomorphism/hand_index.h>
#include <pluribus/poker.hpp>
#include <pluribus/actions.hpp>

namespace pluribus {

class HandIndexer {
public:
  hand_index_t index(const uint8_t cards[], int round) { return hand_index_last(&_indexers[round], cards); }
  hand_index_t index(const Board& board, const Hand& hand, int round);

  hand_indexer_t* get_indexer(int round) { return &_indexers[round]; }

  static HandIndexer* get_instance() {
    if(!_instance) {
      _instance = std::unique_ptr<HandIndexer>(new HandIndexer());
    }
    return _instance.get();
  }

  HandIndexer(const HandIndexer&) = delete;
  HandIndexer& operator=(const HandIndexer&) = delete;

private:
  HandIndexer();

  std::array<hand_indexer_t, 4> _indexers;

  static std::unique_ptr<HandIndexer> _instance;
};

class CachedIndexer {
public:
  CachedIndexer(int max_round = 3);

  hand_index_t index(const uint8_t cards[], int round);
  hand_index_t index(const Board& board, const Hand& hand, int round);
private:
  hand_indexer_state_t _state;
  std::vector<hand_index_t> _indices;
  int _max_round;
};

long count_infosets(const PokerState& state, const ActionProfile& action_profile, int max_round = 4);
long count_actionsets(const PokerState& state, const ActionProfile& action_profile, int max_round = 4);

}
