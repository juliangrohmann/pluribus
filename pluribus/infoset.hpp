#pragma once

#include <hand_isomorphism/hand_index.h>
#include <pluribus/actions.hpp>

namespace pluribus {

class InformationSet {
public:
  InformationSet(const ActionHistory& history, const Board& board, const Hand& hand, int round, int n_players, int n_chips, int ante);
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
  static const std::array<hand_indexer_t, 4> _indexers;
  static const std::array<std::vector<uint16_t>, 4> _cluster_map;
  int _history_idx;
  uint16_t _cluster;
};

long count_infosets(const PokerState& state, int max_round = 4);
long count_actionsets(const PokerState& state, int max_round = 4);

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
