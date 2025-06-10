#include <iostream>
#include <iterator>
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
  struct sysinfo info;
  if(sysinfo(&info) != 0) {
    std::cerr << "getFreeRAM --- sysinfo failed" << std::endl;
    return -1; // Error
  }
  // freeram is in bytes, adjusted by mem_unit
  return static_cast<long long>(info.freeram) * info.mem_unit;
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

std::vector<std::string> get_filepaths(std::string path) {
  std::filesystem::path base_dir{path};
  std::vector<std::string> fns;
  for(const auto& entry : std::filesystem::directory_iterator(path)) {
    fns.push_back((base_dir / entry.path().filename().string()).string());
  }
  return fns;
}

void write_to_file(const std::filesystem::path& file_path, const std::string& content) {
  std::ofstream out_file(file_path, std::ios::out | std::ios::trunc);
  if(!out_file.is_open()) throw std::runtime_error("Failed to open or create file: " + file_path.string());
  out_file << content;
  out_file.close();
}

std::string date_time_str() {
    std::time_t t = std::time(nullptr);
    std::tm tm;
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
    return oss.str();
}

uint8_t card_to_idx(const std::string& card) {
  assert(card.length() == 2 && "Card string must have length == 2.");
  return omp::RANKS.find(card[0]) * 4 + omp::SUITS.find(card[1]);
}

std::string idx_to_card(int idx) {
  return std::string(1, omp::RANKS[idx / 4]) + omp::SUITS[idx % 4];
}

void str_to_cards(std::string card_str, uint8_t cards[]) {
  for(int i = 0; i < card_str.length(); i += 2) {
    cards[i / 2] = card_to_idx(card_str.substr(i, 2));
  }
}

std::string cards_to_str(const uint8_t* begin, const uint8_t* end) {
  std::string str = "";
  for(; begin != end; ++begin) {
    str += idx_to_card(*begin);
  }
  return str;
}

std::string cards_to_str(const uint8_t cards[], int n) {
  return cards_to_str(cards, cards + n);
}

std::string cards_to_str(const std::vector<uint8_t>& cards) {
  return cards_to_str(&cards[0], &cards[0] + cards.size());
}

int n_board_cards(int round) {
  return round == 0 ? 0 : std::min(2 + round, 5);
}

int init_indexer(hand_indexer_t& indexer, int round) {
  uint8_t n_cards[round + 1];
  uint8_t all_rounds[] = {2, 3, 1, 1};
  int card_sum = 0;
  for(int i = 0; i < round + 1; ++i) {
    n_cards[i] = all_rounds[i];
    card_sum += all_rounds[i];
  }
  bool init_success = hand_indexer_init(round + 1, n_cards, &indexer);
  assert(init_success && "Failed to initialize indexer.");
  return card_sum;
}

}