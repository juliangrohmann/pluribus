#pragma once

#include <unordered_map>
#include <pluribus/actions.hpp>
#include <pluribus/poker.hpp>

namespace pluribus {

// using HistoryMap = std::unordered_map<ActionHistory, int>;

// class HistoryIndexer {
// public:
//   void initialize(const PokerConfig& config);
//   int index(const ActionHistory& history, const PokerConfig& config);
//   size_t size(const PokerConfig& config);

//   static HistoryIndexer* get_instance() {
//     if(!_instance) {
//       _instance = std::unique_ptr<HistoryIndexer>(new HistoryIndexer());
//     }
//     return _instance.get();
//   }

//   HistoryIndexer(const HistoryIndexer&) = delete;
//   HistoryIndexer& operator==(const HistoryIndexer&) = delete;
// private:
//   HistoryIndexer() {}

//   std::unordered_map<std::string, HistoryMap> _history_map;

//   static std::unique_ptr<HistoryIndexer> _instance;
// };

// std::string history_map_filename(const PokerConfig& config);
// HistoryMap build_history_map(const PokerState& state, const ActionProfile& action_profile);

}