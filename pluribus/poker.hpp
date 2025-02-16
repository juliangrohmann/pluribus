#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <initializer_list>
#include <hand_isomorphism/hand_index.h>
#include <omp/Hand.h>
#include <omp/HandEvaluator.h>
#include <boost/dynamic_bitset.hpp>
#include <pluribus/actions.hpp>

namespace pluribus {

class Deck {
public:
  Deck() { reset(); }
  inline int draw() { return _cards[_current++]; }
  void reset();
  void shuffle();

private:
    std::array<uint8_t, 52> _cards;
    uint8_t _current = 0;
    std::mt19937 _rng{ std::random_device{}() };
};

class Player {
public:
  Player(int chips) : _chips{chips} {};
  Player(const Player&) = default;
  Player(Player&&) = default;
  Player& operator=(const Player&) = default;
  Player& operator=(Player&&) = default;
  inline int get_chips() const { return _chips; }
  inline int get_betsize() const { return _betsize; }
  inline bool has_folded() const { return _folded; }
  void invest(int amount);
  void next_round();
  void fold();
  void reset(int chips);

private:
  int _chips;
  int _betsize = 0;
  bool _folded = false;
};

class PokerState {
public:
  PokerState(int n_players, int chips, int ante);
  PokerState(const PokerState&) = default;
  PokerState(PokerState&&) = default;
  PokerState& operator=(const PokerState&) = default;
  PokerState& operator=(PokerState&&) = default;
  inline const std::vector<Player>& get_players() const { return _players; }
  inline int get_pot() const { return _pot; }
  inline int get_max_bet() const { return _max_bet; }
  inline uint8_t get_active() const { return _active; }
  inline uint8_t get_round() const { return _round; }
  inline uint8_t get_bet_level() const { return _bet_level; }
  inline int8_t get_winner() const { return _winner; }
  PokerState apply(Action action) const;
  
private:
  std::vector<Player> _players;
  ActionHistory _actions;
  int _pot;
  int _max_bet;
  short _action_counter;
  uint8_t _active;
  uint8_t _round;
  uint8_t _bet_level;
  int8_t _winner;

  PokerState bet(int amount) const;
  PokerState call() const;
  PokerState check() const;
  PokerState fold() const;
  PokerState next_state(Action action) const;
  void next_player();
  void next_round();
};

int total_bet_size(const PokerState& state, Action action);
std::vector<Action> all_actions(const PokerState& state);
std::vector<Action> valid_actions(const PokerState& state);

std::vector<uint8_t> winners(const PokerState& state, const std::vector<std::array<uint8_t, 2>>& hands, 
               const std::array<uint8_t, 5>& board_cards, const omp::HandEvaluator& eval);
void deal_hands(Deck& deck, std::vector<std::array<uint8_t, 2>>& hands);
void deal_board(Deck& deck, std::array<uint8_t, 5>& board);
}
