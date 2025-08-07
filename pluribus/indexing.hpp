#pragma once

#include <hand_isomorphism/hand_index.h>
#include <pluribus/actions.hpp>
#include <pluribus/poker.hpp>

namespace pluribus {

class HandIndexer {
public:
  hand_index_t index(const uint8_t cards[], const int round) const { return hand_index_last(&_indexers[round], cards); }
  hand_index_t index(const Board& board, const Hand& hand, int round) const;
  void unindex(const hand_index_t index, uint8_t cards[], const int round) const { hand_unindex(&_indexers[round], round, index, cards); }
  hand_index_t size(const int round) const { return hand_indexer_size(&_indexers[round], round); }

  hand_indexer_t* get_indexer(const int round) { return &_indexers[round]; }

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

class FlopIndexer {
public:
  hand_index_t index(const uint8_t flop[]) const { return hand_index_last(&_indexer, flop); }
  hand_index_t index(const Board& board) const { return index(board.cards().data()); }
  void unindex(const hand_index_t index, uint8_t cards[]) const { hand_unindex(&_indexer, 0, index, cards); }
  static FlopIndexer* get_instance() {
    if(!_instance) {
      _instance = std::unique_ptr<FlopIndexer>(new FlopIndexer());
    }
    return _instance.get();
  }

  FlopIndexer(const FlopIndexer&) = delete;
  FlopIndexer& operator=(const FlopIndexer&) = delete;

private:
  FlopIndexer();

  hand_indexer_t _indexer;

  static std::unique_ptr<FlopIndexer> _instance;
};

class CachedIndexer {
public:
  explicit CachedIndexer(int max_round = 3);

  hand_index_t index(const uint8_t cards[], int round);
  hand_index_t index(const Board& board, const Hand& hand, int round);
private:
  hand_indexer_state_t _state{};
  std::vector<hand_index_t> _indices;
  int _max_round;
};

long count_infosets(const PokerState& state, const ActionProfile& action_profile, int max_round = 4);
long count_actionsets(const PokerState& state, const ActionProfile& action_profile, int max_round = 4);

}
