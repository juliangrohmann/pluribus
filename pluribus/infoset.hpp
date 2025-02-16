#pragma once

#include <pluribus/actions.hpp>

namespace pluribus {

class InformationSet {
public:
const ActionHistory& getHistory() const { return _history; }
  uint16_t getCluster() const { return cluster; }
private:
  ActionHistory _history;
  uint16_t cluster;
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
