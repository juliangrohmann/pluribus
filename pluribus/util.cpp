#include <iostream>
#include <iomanip>
#include <cassert>
#include <string>
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <sys/sysinfo.h>
#include <omp/Hand.h>
#include <hand_isomorphism/hand_index.h>

namespace pluribus {

long long get_free_ram() {
  std::ifstream meminfo("/proc/meminfo");
  if(!meminfo.is_open()) {
    std::cerr << "get_free_ram --- Failed to open /proc/meminfo" << std::endl;
    return -1;
  }

  std::string line;
  while(std::getline(meminfo, line)) {
    if(line.find("MemAvailable:") == 0) {
      std::istringstream iss(line);
      std::string label;
      long long kb;
      std::string unit;
      iss >> label >> kb >> unit;
      return kb * 1024;
    }
  }

  std::cerr << "get_free_ram --- MemAvailable not found in /proc/meminfo" << std::endl;
  return -1;
}

bool create_dir(const std::filesystem::path& path) {
  try {
    if(std::filesystem::exists(path)) {
        return true;
    }
    return std::filesystem::create_directory(path);
  }
  catch(const std::exception& e) {
    std::cerr << "Error creating directory: " << e.what() << std::endl;
    return false;
  }
}

std::vector<std::string> get_filepaths(const std::string& path) {
  const std::filesystem::path base_dir{path};
  std::vector<std::string> fns;
  for(const auto& entry : std::filesystem::directory_iterator(path)) {
    fns.push_back((base_dir / entry.path().filename().string()).string());
  }
  return fns;
}

void write_to_file(const std::filesystem::path& file_path, const std::string& content, const bool append) {
  const auto mode = std::ios::out | (append ? std::ios::app : std::ios::trunc);
  std::ofstream out_file(file_path, mode);
  if(!out_file.is_open()) throw std::runtime_error("Failed to open or create file: " + file_path.string());
  out_file << content;
  out_file.close();
}

std::string date_time_str(const std::string& format) {
    std::time_t t = std::time(nullptr);
    std::tm tm;
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, format.c_str());
    return oss.str();
}

uint8_t card_to_idx(const std::string& card) {
  assert(card.length() == 2 && "Card string must have length == 2.");
  return omp::RANKS.find(card[0]) * 4 + omp::SUITS.find(card[1]);
}

std::string idx_to_card(const int idx) {
  return std::string(1, omp::RANKS[idx / 4]) + omp::SUITS[idx % 4];
}

void str_to_cards(const std::string& card_str, uint8_t cards[]) {
  for(int i = 0; i < card_str.length(); i += 2) {
    cards[i / 2] = card_to_idx(card_str.substr(i, 2));
  }
}

std::vector<uint8_t> str_to_cards(const std::string& card_str) {
  std::vector<uint8_t> cards(card_str.length() / 2);
  str_to_cards(card_str, cards.data());
  return cards;
}

std::string cards_to_str(const uint8_t* begin, const uint8_t* end) {
  std::string str = "";
  for(; begin != end; ++begin) {
    str += idx_to_card(*begin);
  }
  return str;
}

std::string cards_to_str(const uint8_t cards[], const int n) {
  return cards_to_str(cards, cards + n);
}

std::string cards_to_str(const std::vector<uint8_t>& cards) {
  return cards_to_str(&cards[0], &cards[0] + cards.size());
}

int n_board_cards(const int round) {
  return round == 0 ? 0 : std::min(2 + round, 5);
}

int init_indexer(hand_indexer_t& indexer, const int round) {
  uint8_t n_cards[round + 1];
  constexpr uint8_t all_rounds[] = {2, 3, 1, 1};
  int card_sum = 0;
  for(int i = 0; i < round + 1; ++i) {
    n_cards[i] = all_rounds[i];
    card_sum += all_rounds[i];
  }
  bool init_success = hand_indexer_init(round + 1, n_cards, &indexer);
  assert(init_success && "Failed to initialize indexer.");
  return card_sum;
}

std::string join_strs(const std::vector<std::string>& strs, const std::string& sep) {
  std::ostringstream oss;
  for(int i = 0; i < strs.size(); ++i) oss << strs[i] << (i == strs.size() - 1 ? "" : sep);
  return oss.str();
}

}