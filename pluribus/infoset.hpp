#pragma once

#include <hand_isomorphism/hand_index.h>
#include <pluribus/actions.hpp>

namespace pluribus {

class InformationSet {
public:
  InformationSet(const ActionHistory& history, const Board& board, const Hand& hand, int round);
  InformationSet() = default;
  bool operator==(const InformationSet& other) const;
  const ActionHistory& getHistory() const { return _history; }
  uint16_t getCluster() const { return _cluster; }
  std::string to_string() const;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_history, _cluster);
  }
private:
  static const std::array<hand_indexer_t, 4> _indexers;
  static const std::array<std::vector<uint16_t>, 4> _cluster_map;
  ActionHistory _history;
  uint16_t _cluster;
};

}

namespace std {

template <>
struct hash<pluribus::InformationSet> {
  std::size_t operator()(const pluribus::InformationSet &info) const {
      std::size_t h1 = std::hash<pluribus::ActionHistory>{}(info.getHistory());
      std::size_t h2 = std::hash<uint16_t>{}(info.getCluster());
      return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
  }
};

}
