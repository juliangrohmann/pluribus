#pragma once

#include <string>
#include <vector>
#include <string>
#include <unordered_map>

class Deck {
  public:
    Deck() {
      
    };
    int draw();
    void reset();

  private:
    std::vector<int> cards;
};

class Evaluator {
  public:
    Evaluator();
    int eval(const std::array<int, 2>& hand, const std::array<int, 5>& board) const;
    int eval(const std::array<int, 5>& cards) const;
  private:
    std::unordered_map<int, int> flush_lut;
    std::unordered_map<int, int> unsuited_lut;

    void cache_unpaired();
    void cache_paired();
};

int make_card(const std::string& card);
std::array<int, 2> make_hand(const std::string& str);
std::array<int, 5> make_board(const std::string& str);
std::string decode_cards(int card);
template <std::size_t N>
std::string decode_cards(const std::array<int, N>& hand) {
  std::string ret = "";
  for(int card : hand) {
    ret += decode_cards(card);
  }
  return ret;
}
