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
  if(size == 73346583400) {
    std::cout << "Skipping regrets\n";
    return;
  } 
  const size_t CHUNK = (size_t(1) * 1024 * 1024) / sizeof(T);
  size_t loaded = 0;
  while(loaded < size) {
    size_t this_chunk = std::min(CHUNK, size - loaded);
    std::cout << "Size (bef)=" << vec.size() << "\n";
    std::cout << "Growing to " << loaded + this_chunk << "\n";
    vec.grow_to_at_least(loaded + this_chunk);
    std::cout << "Grew to    " << loaded + this_chunk << "\n";
    std::cout << "Size (aft)=" << vec.size() << "\n";
    std::cout << "Loaded (before) = " << loaded << '\n';
    if(loaded + this_chunk - 1 >= vec.size()) throw std::runtime_error("Overflow. Size=" + std::to_string(vec.size()) + ", Idx=" + std::to_string(loaded + this_chunk - 1));
    for(size_t i = 0; i < this_chunk; ++i) {
      ar(vec[loaded + i]);
    }
    std::cout << "Loaded (after)  = " << loaded << '\n';
    std::cout << "Final size=" << size << "\n";
    std::cout << "-----------------------------\n";
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