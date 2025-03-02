#include <pluribus/storage.hpp>

namespace pluribus {

RegretStorage::RegretStorage(int n_players, int n_chips, int ante, int n_clusters, int n_actions) : _n_clusters{n_clusters}, _n_actions{n_actions} {
  HistoryIndexer::get_instance()->initialize(n_players, n_chips, ante);
  std::cout << "Constructing regret storage... " << std::flush;
  size_t n_histories = HistoryIndexer::get_instance()->size(n_players, n_chips, ante);
  _size = n_histories * n_clusters * n_actions;
  _fn = "atomic_regrets_XXXXXX";
  _data = map_memory<std::atomic<int>>(_size, _fn, _fd);

  std::cout << "Initializing regrets... " << std::flush;
  #pragma omp parallel for schedule(static, 1)
  for(size_t i = 0; i < _size; ++i) {
    (_data + i)->store(0, std::memory_order_relaxed);
  }
  std::cout << "Success.\n";
}

RegretStorage::~RegretStorage() {
  unmap_memory(_data, _size, _fn, _fd);
}

std::atomic<int>* RegretStorage::operator[](const InformationSet& info_set) {
  return _data + info_offset(info_set);
}

const std::atomic<int>* RegretStorage::operator[](const InformationSet& info_set) const {
  return _data + info_offset(info_set);
}

bool RegretStorage::operator==(const RegretStorage& other) const {
  if(_size != other._size || _n_clusters != other._n_clusters || _n_actions != other._n_actions) return false;
  for(size_t i = 0; i < _size; ++i) {
    if(*(_data + i) != *(other._data + i)) return false;
  }
  return true;
}

size_t RegretStorage::info_offset(const InformationSet& info_set) const {
  return (static_cast<size_t>(info_set.get_history_idx()) * _n_clusters + info_set.get_cluster()) * _n_actions;
}

ActionStorage::ActionStorage(int n_players, int n_chips, int ante, int n_clusters) : _n_clusters{n_clusters} {
  HistoryIndexer::get_instance()->initialize(n_players, n_chips, ante);
  std::cout << "Constructing action storage... " << std::flush;
  size_t n_histories = HistoryIndexer::get_instance()->size(n_players, n_chips, ante);
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