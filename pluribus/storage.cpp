#include <pluribus/storage.hpp>
#include <stdexcept>
#include <pluribus/history_index.hpp>

namespace pluribus {

size_t RegretStorage::index(const PokerState& state, int cluster, int action) {
  auto actions = valid_actions(state, _action_profile);
  size_t n_actions = actions.size();
  auto history = state.get_action_history();

  size_t history_idx;
  auto it = _history_map.find(history);
  if (it != _history_map.end()) {
    history_idx = it->second;
  } else {
    std::lock_guard<std::mutex> lock(_grow_mutex);
    it = _history_map.find(history); // Double-check
    if (it == _history_map.end()) {
      history_idx = _data.size();
      _history_map.insert({history, history_idx});
      _data.grow_by(_n_clusters * n_actions);
    } else {
      history_idx = it->second;
    }
  }

  // Ensure growth is complete before returning
  size_t max_idx = history_idx + _n_clusters * n_actions - 1;
  while (max_idx >= _data.size()) {
    std::this_thread::yield(); // Wait for growth if another thread is still growing
  }
  (void)_data[max_idx].load(std::memory_order_acquire); // Ensure construction is visible

  return history_idx + cluster * n_actions + action;
}

size_t RegretStorage::index(const PokerState& state, int cluster, int action) const {
  size_t n_actions = valid_actions(state, _action_profile).size();
  auto it = _history_map.find(state.get_action_history());
  if(it != _history_map.end()) return it->second + cluster * n_actions + action;
  throw std::runtime_error("RegretStorage --- Indexed out of range.");
}

bool RegretStorage::operator==(const RegretStorage& other) const {
  return _data == other._data &&
         _history_map == other._history_map &&
         _action_profile == other._action_profile &&
         _n_clusters == other._n_clusters;
}

ActionStorage::ActionStorage(const PokerConfig& config, int n_clusters) : _n_clusters{n_clusters} {
  HistoryIndexer::get_instance()->initialize(config);
  std::cout << "Constructing action storage... " << std::flush;
  size_t n_histories = HistoryIndexer::get_instance()->size(config);
  _size = n_histories * n_clusters;
  _fn = "action_storage_XXXXXX";
  _data = map_memory<Action>(_size, _fn, _fd);

  std::cout << "Initializing actions... " << std::flush;
  #pragma omp parallel for schedule(static, 1)
  for(size_t i = 0; i < _size; ++i) {
    *(_data + i) = Action::UNDEFINED;
  }
  std::cout << "Success.\n";
  std::cout << "_size=" << _size << "\n";
}

ActionStorage::~ActionStorage() {
  unmap_memory(_data, _size, _fn, _fd);
}

Action& ActionStorage::operator[](const InformationSet& info_set) {
  return _data[info_offset(info_set)];
}

const Action& ActionStorage::operator[](const InformationSet& info_set) const {
  return _data[info_offset(info_set)];
}

size_t ActionStorage::info_offset(const InformationSet& info_set) const {
  return static_cast<size_t>(info_set.get_history_idx()) * _n_clusters + info_set.get_cluster();
}

}