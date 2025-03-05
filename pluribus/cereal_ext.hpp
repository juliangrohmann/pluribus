#pragma once

#include <atomic>
#include <cereal/cereal.hpp>
#include <tbb/concurrent_vector.h>
#include <tbb/concurrent_unordered_map.h>

namespace pluribus {
  
template <class T>
void cereal_save(const T& data, const std::string& fn) {
  std::cout << "Saving to " << fn << '\n';
  std::ofstream os(fn, std::ios::binary);
  cereal::BinaryOutputArchive oarchive(os);
  oarchive(data);
}

template <class T>
T cereal_load(const std::string& fn) {
  std::cout << "Loading from " << fn << '\n';
  std::ifstream is(fn, std::ios::binary);
  cereal::BinaryInputArchive iarchive(is);
  T data;
  iarchive(data);
  return data;
}

}

namespace cereal {

template<class Archive>
void serialize(Archive& archive, std::atomic<int>& atomic_int) {
    int value = atomic_int.load(std::memory_order_relaxed);
    archive(value);
    if(Archive::is_loading::value) {
      atomic_int.store(value, std::memory_order_relaxed);
    }
}

template<class Archive, class T>
void save(Archive& ar, const tbb::concurrent_vector<T>& vec) {
  size_t size = vec.size();
  ar(size);
  for(size_t i = 0; i < size; ++i) {
    ar(vec[i]);
  }
}

template<class Archive, class T>
void load(Archive& ar, tbb::concurrent_vector<T>& vec) {
  size_t size;
  ar(size);
  vec.clear();
  auto it = vec.grow_by(size);
  for(size_t i = 0; i < size; ++i) {
    ar(*it);
    ++it;
  }
}

template<class Archive, class Key, class T>
void save(Archive& ar, const tbb::concurrent_unordered_map<Key, T>& map) {
  size_t size = map.size();
  ar(size);
  for(const auto& pair : map) {
    ar(pair.first, pair.second);
  }
}

template<class Archive, class Key, class T>
void load(Archive& ar, tbb::concurrent_unordered_map<Key, T>& map) {
  size_t size;
  ar(size);
  map.clear();
  for(size_t i = 0; i < size; ++i) {
    Key key;
    T value;
    ar(key, value);
    map[key] = value;
  }
}

}