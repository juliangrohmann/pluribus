#include <pluribus/history_index.hpp>
#include <pluribus/cereal_ext.hpp>

namespace pluribus {

// std::unique_ptr<HistoryIndexer> HistoryIndexer::_instance = nullptr;

// void HistoryIndexer::initialize(const PokerConfig& config) {
//   std::string fn = history_map_filename(config);
//   if(_history_map.find(fn) == _history_map.end()) {
//     std::cout << "Initializing history map (n_players=" << config.n_players << ", n_chips=" << config.n_chips << ", ante=" << config.ante 
//               << ")... " << std::flush;
//     _history_map[fn] = cereal_load<HistoryMap>(fn);
//     std::cout << "Success.\n";
//   }
// }

// int HistoryIndexer::index(const ActionHistory& history, const PokerConfig& config) {
//   std::string fn = history_map_filename(config);
//   auto it = _history_map.find(fn);
//   if(it == _history_map.end()) {
//     initialize(config);
//     return _history_map.at(fn).at(history);
//   }
//   return it->second.at(history);
// }

// size_t HistoryIndexer::size(const PokerConfig& config) {
//   std::string fn = history_map_filename(config);
//   auto it = _history_map.find(fn);
//   if(it != _history_map.end()) {
//     return it->second.size();
//   }
//   else {
//     throw std::runtime_error("HistoryIndexer::size --- HistoryIndexer is uninitialized.");
//   }
// }

// void collect_histories(const PokerState& state, std::vector<ActionHistory>& histories, const ActionProfile& action_profile) {
//   if(state.is_terminal()) return;
//   histories.push_back(state.get_action_history());
//   for(Action a : valid_actions(state, action_profile)) {
//     collect_histories(state.apply(a), histories, action_profile);
//   }
// }

// std::string history_map_filename(const PokerConfig& config) {
//   return "history_index_p" + std::to_string(config.n_players) + "_" + std::to_string(config.n_chips / 100) + "bb_" + 
//          std::to_string(config.ante) + "ante.bin";
// }

// HistoryMap build_history_map(const PokerState& state, const ActionProfile& action_profile) {
//   std::cout << "Building history map:\n";
//   std::cout << action_profile.to_string();
//   std::vector<ActionHistory> histories;
//   collect_histories(state, histories, action_profile);
//   HistoryMap history_map;
//   for(long i = 0; i < histories.size(); ++i) {
//     history_map[histories[i]] = i;
//   }
//   std::cout << "n_histories=" << histories.size() << "\n";
//   return history_map;
// }

}