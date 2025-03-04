#pragma once

#include <hand_isomorphism/hand_index.h>
#include <pluribus/poker.hpp>
#include <pluribus/actions.hpp>

namespace pluribus {

class HandIndexer {
public:
  uint64_t index(const uint8_t cards[], int round) { return hand_index_last(&_indexers[round], cards); }

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

class InformationSet {
public:
  InformationSet(const ActionHistory& history, const Board& board, const Hand& hand, int round, const PokerConfig& config);
  InformationSet(const ActionHistory& history, uint16_t cluster, const PokerConfig& config);
  InformationSet() = default;
  bool operator==(const InformationSet& other) const;
  int get_history_idx() const { return _history_idx; }
  uint16_t get_cluster() const { return _cluster; }
  std::string to_string() const;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_history_idx, _cluster);
  }
private:
  int _history_idx;
  uint16_t _cluster;
};

long count_infosets(const PokerState& state, const ActionProfile& action_profile, int max_round = 4);
long count_actionsets(const PokerState& state, const ActionProfile& action_profile, int max_round = 4);

}

namespace std {

template <>
struct hash<pluribus::InformationSet> {
  std::size_t operator()(const pluribus::InformationSet &info) const {
      std::size_t h1 = std::hash<int>{}(info.get_history_idx());
      std::size_t h2 = std::hash<uint16_t>{}(info.get_cluster());
      return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
  }
};

}
