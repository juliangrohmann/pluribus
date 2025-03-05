#pragma once

#include <atomic>
#include <iostream>
#include <filesystem>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_vector.h>
#include <omp.h>
#include <pluribus/cereal_ext.hpp>
#include <pluribus/infoset.hpp>
#include <pluribus/history_index.hpp>
#include <pluribus/actions.hpp>

namespace pluribus {

template <class T>
T* map_memory(size_t sz, std::string& filename_temp, int& fd) {
  std::cout << "Opening map file... " << std::flush;
  char fn_temp[] = "temp_XXXXXX";
  fd = mkstemp(fn_temp);
  if(fd == -1) {
    throw std::runtime_error("Failed to create map file.");
  }
  filename_temp = fn_temp;
  std::cout << "Opened " + filename_temp + "... " << std::flush;

  size_t file_size = sz * sizeof(T);
  std::cout << "Resizing file to " << file_size << " bytes... " << std::flush;
  if(ftruncate(fd, file_size) == -1) {
    close(fd);
    throw std::runtime_error("Failed to resize file.");
  }

  std::cout << "Mapping file... ";
  void* ptr = mmap(NULL, sz * sizeof(T), PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
  if(ptr == MAP_FAILED) {
    close(fd);
    throw std::runtime_error("Failed to map file to memory.");
  }
  std::cout << "Mapped.\n";
  return static_cast<T*>(ptr);
}

template <class T>
void unmap_memory(T* data, size_t sz, const std::string& filename, int fd) {
  std::cout << "Unmapping memory... " << std::flush;
  if(data) munmap(data, sz * sizeof(T));
  if(fd != -1) close(fd);
  if(unlink(filename.c_str()) != 0) {
    std::cerr << "Failed to delete file " << filename << ": " << strerror(errno) << "\n";
  } 
  else {
    std::cout << "Unmapped.\n";
  }
}

class RegretStorage {
public:
  RegretStorage(const ActionProfile& action_profile = BlueprintActionProfile{}, int n_clusters = 200) : 
               _action_profile{action_profile}, _n_clusters{n_clusters} {};

  inline const tbb::concurrent_vector<std::atomic<int>>& data() const { return _data; }
  inline tbb::concurrent_vector<std::atomic<int>>& data() { return _data; }
  inline int get_n_clusters() const { return _n_clusters; }

  std::atomic<int>& operator[](size_t idx) { return _data[idx]; }
  const std::atomic<int>& operator[](size_t idx) const { return _data[idx]; }
  size_t index(const PokerState& state, int cluster, int action = 0);
  size_t index(const PokerState& state, int cluster, int action = 0) const;
  bool operator==(const RegretStorage& other) const;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_data, _history_map, _action_profile, _n_clusters);
  }

private:
  tbb::concurrent_vector<std::atomic<int>> _data;
  tbb::concurrent_unordered_map<ActionHistory, int> _history_map;
  ActionProfile _action_profile;
  int _n_clusters;
  std::mutex _grow_mutex;
};

class ActionStorage {
public:
  ActionStorage(const PokerConfig& config = PokerConfig{}, int n_clusters = 200);
  ~ActionStorage();
  Action& operator[](const InformationSet& info_set);
  const Action& operator[](const InformationSet& info_set) const;
  inline Action* data() { return _data; }
  inline const Action* data() const { return _data; }
  inline size_t size() { return _size; }
  inline int get_n_clusters() const { return _n_clusters; }
  
private:
  size_t info_offset(const InformationSet& info_set) const;
  
  Action* _data;
  size_t _size;
  std::string _fn;
  int _n_clusters;
  int _fd;
};

}