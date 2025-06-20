#pragma once

#include <atomic>
#include <cereal/cereal.hpp>
#include <tbb/concurrent_vector.h>
#include <tbb/concurrent_unordered_map.h>
#include <pluribus/logging.hpp>

namespace pluribus {
  
template <class T>
void cereal_save(const T& data, const std::string& fn) {
  Logger::log("Saving to " + fn + '\n');
  std::ofstream os(fn, std::ios::binary);
  cereal::BinaryOutputArchive oarchive(os);
  oarchive(data);
  Logger::log("Saved successfully.");
}

template <class T>
void cereal_load(T& data, const std::string& fn) {
  Logger::log("Loading from " + fn + '\n');
  std::ifstream is(fn, std::ios::binary);
  cereal::BinaryInputArchive iarchive(is);
  iarchive(data);
  Logger::log("Loaded successfully.");
}

}

namespace cereal {

template<class Archive, class T>
void serialize(Archive& ar, std::atomic<T>& atomic) {
  T value = Archive::is_loading::value ? T{} : atomic.load();
  ar(value);
  if (Archive::is_loading::value) {
    atomic.store(value);
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

// template<class Archive, class T>
// void load(Archive& ar, tbb::concurrent_vector<T>& vec) {
//   size_t size;
//   ar(size);
//   vec.clear();

//   const size_t CHUNK = 10UL * 1024 * 1024 * 1024 / sizeof(T);
//   size_t remaining = size;
//   while(remaining > 0) {
//     size_t this_chunk = std::min(CHUNK, remaining);
//     auto it = vec.grow_by(this_chunk);
//     for(size_t i = 0; i < this_chunk; ++i, ++it) {
//       ar(*it);
//     }
//     remaining -= this_chunk;
//   }
// }

template<class Archive, class T>
void load(Archive& ar, tbb::concurrent_vector<T>& vec) {
  size_t size;
  ar(size);
  vec.clear();
  const size_t CHUNK = (size_t(1) * 1024 * 1024 * 1024) / sizeof(T);
  size_t loaded = 0;
  while(loaded < size) {
    size_t this_chunk = std::min(CHUNK, size - loaded);
    vec.grow_to_at_least(loaded + this_chunk);
    for(size_t i = 0; i < this_chunk; ++i) {
      ar(vec[loaded + i]);
    }
    loaded += this_chunk;
  }
  std::cout << "Successful concurrent vector load.\n";
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