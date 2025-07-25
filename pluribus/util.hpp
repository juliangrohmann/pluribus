#pragma once

#include <string>
#include <fstream>
#include <filesystem>
#include <json/json.hpp>
#include <hand_isomorphism/hand_index.h>

namespace pluribus {

long long get_free_ram();
bool create_dir(const std::filesystem::path& path);
std::vector<std::string> get_filepaths(const std::string &path);
void write_to_file(const std::filesystem::path& file_path, const std::string& content, bool append = false);
std::string date_time_str(const std::string& format = "%Y-%m-%d_%H-%M-%S");
uint8_t card_to_idx(const std::string& card);
std::string idx_to_card(int idx);
void str_to_cards(const std::string &card_str, uint8_t cards[]);
std::vector<uint8_t> str_to_cards(const std::string &card_str);
std::string cards_to_str(const uint8_t cards[], int n);
std::string cards_to_str(const std::vector<uint8_t>& cards);
int n_board_cards(int round);
int init_indexer(hand_indexer_t& indexer, int round);
std::string join_strs(const std::vector<std::string>& strs, const std::string& sep);

template <class T>
std::string join_as_strs(const std::vector<T>& vals, const std::string& sep) {
  std::ostringstream oss;
  for(int i = 0; i < vals.size(); ++i) oss << std::to_string(vals[i]) << (i == vals.size() - 1 ? "" : sep);
  return oss.str();
}

template <class T> 
int index_of(T e, const std::vector<T>& v) {
  auto it = std::find(v.begin(), v.end(), e);
  if(it == v.end()) throw std::runtime_error("Failed to find element in vector of " + std::to_string(v.size()) + " elements."); 
  return std::distance(v.begin(), it);
}

}