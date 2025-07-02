#pragma once

#include <string>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <initializer_list>
#include <hand_isomorphism/hand_index.h>
#include <omp/Hand.h>
#include <omp/HandEvaluator.h>
#include <cereal/types/array.hpp>
#include <pluribus/constants.hpp>
#include <pluribus/util.hpp>
#include <pluribus/actions.hpp>

namespace pluribus {

class Deck {
public:
  Deck(const std::unordered_set<uint8_t>& dead_cards = {}) : _dead_cards{dead_cards} { reset(); }
  Deck(const std::vector<uint8_t>& dead_cards) {
    std::copy(dead_cards.begin(), dead_cards.end(), std::inserter(_dead_cards, _dead_cards.end()));
    reset();
  }

  int draw();
  void add_dead_card(uint8_t card) { _dead_cards.insert(card); }

  template<typename Container>
  void add_dead_cards(const Container& cont) { for(auto it = cont.begin(); it != cont.end(); ++it) add_dead_card(*it); }

  void clear_dead_cards() { _dead_cards.clear(); }
  void reset();
  void shuffle();

private:
    std::array<uint8_t, MAX_CARDS> _cards;
    std::unordered_set<uint8_t> _dead_cards;
    uint8_t _current = 0;
};

inline uint64_t card_mask(uint8_t card) { return 1L << card; }
uint64_t card_mask(const std::vector<uint8_t>& cards);

template<int N>
class CardSet {
public:
  CardSet() { 
    _cards.fill(0); 
    update_mask();
  }
  CardSet(const std::array<uint8_t, N>& cards) : _cards{cards} { 
    update_mask(); 
  }
  CardSet(const std::vector<uint8_t>& cards) { 
    std::copy(cards.begin(), cards.end(), _cards.begin()); 
    update_mask(0, cards.size()); 
  }
  CardSet(const std::initializer_list<uint8_t>& cards) { 
    std::copy(cards.begin(), cards.end(), _cards.begin()); 
    update_mask(0, cards.size()); 
  }
  CardSet(const std::string& card_str) { 
    str_to_cards(card_str, _cards.data()); 
    update_mask(0, card_str.size() / 2); 
  }
  CardSet(Deck& deck, const std::vector<uint8_t> init_cards = {}) { 
    deal(deck, init_cards); 
  }

  void set_card(int i, uint8_t card) {
    _cards[i] = card;
    update_mask(i, i + 1);
  }

  bool collides(const CardSet<N>& other) const {
    return _mask & other._mask;
  }

  void deal(Deck& deck, std::vector<uint8_t> init_cards = {}) { 
    for(int i = 0; i < init_cards.size(); ++i) _cards[i] = init_cards[i];
    for(int i = init_cards.size(); i < N; ++i) _cards[i] = deck.draw(); 
    update_mask();
  }

  uint64_t mask() const { return _mask; }
  const std::array<uint8_t, N>& cards() const { return _cards; }
  std::vector<uint8_t> as_vector(int n = -1) const { 
    return std::vector<uint8_t>{_cards.data(), _cards.data() + (n == -1 ? _cards.size() : n)}; 
  }
  std::string to_string() const { return cards_to_str(_cards.data(), N); }

  bool operator==(const CardSet<N>&) const = default;

protected:
  void update_mask(int i = 0, int n = N) {
    _mask = 0L;
    for(int i = 0; i < n; ++i) _mask |= card_mask(_cards[i]);
  }

  std::array<uint8_t, N> _cards;
  uint64_t _mask;
};

class Board : public CardSet<5> {
public:
  Board() : CardSet<5>{} {}
  Board(const std::array<uint8_t, 5>& cards) : CardSet<5>{cards} {}
  Board(const std::vector<uint8_t>& cards) : CardSet<5>{cards} {}
  Board(const std::initializer_list<uint8_t>& cards) : CardSet<5>{cards} {}
  Board(const std::string& card_str) : CardSet<5>{card_str} {};
  Board(Deck& deck, const std::vector<uint8_t>& init_cards = {}) : CardSet<5>{deck, init_cards} {};
  bool operator==(const Board&) const = default;
};

class Hand : public CardSet<2> {
public:
  Hand() : CardSet<2>{} {}
  Hand(const std::array<uint8_t, 2>& cards) : CardSet<2>{cards} {}
  Hand(const std::vector<uint8_t>& cards) : CardSet<2>{cards} {}
  Hand(const std::initializer_list<uint8_t>& cards) : CardSet<2>{cards} {}
  Hand(const std::string& card_str) : CardSet<2>{card_str} {};
  Hand(Deck& deck, const std::vector<uint8_t>& init_cards = {}) : CardSet<2>{deck, init_cards} {};
  bool operator==(const Hand&) const = default;

  static const Hand PLACEHOLDER;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_cards, _mask);
  }
};

inline Hand canonicalize(const Hand& hand) {
  return hand.cards()[0] > hand.cards()[1] ? hand : Hand{hand.cards()[1], hand.cards()[0]};
}

bool collides(uint8_t card, const Hand& hand);
bool collides(uint8_t card, const Board& board);
bool collides(uint8_t card, const std::vector<uint8_t> cards);
bool collides(const Hand& h1, const Hand& h2);
bool collides(const Hand& hand, const Board& board);
bool collides(const Hand& hand, const std::vector<uint8_t>& cards);
std::vector<uint8_t> collect_cards(const Board& board, const Hand& hand, int round = 3);

class Player {
public:
  Player(int chips = 10'000) : _chips{chips} {};
  Player(const Player&) = default;
  Player(Player&&) = default;

  Player& operator=(const Player&) = default;
  Player& operator=(Player&&) = default;
  bool operator==(const Player& other) const = default;

  inline int get_chips() const { return _chips; }
  inline int get_betsize() const { return _betsize; }
  inline bool has_folded() const { return _folded; }
  void invest(int amount);
  void next_round();
  void fold();
  void reset(int chips);

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_chips, _betsize, _folded);
  }

private:
  int _chips;
  int _betsize = 0;
  bool _folded = false;
};

struct PokerConfig {
  std::string to_string() const;

  bool operator==(const PokerConfig&) const = default;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(n_players, n_chips, ante);
  }

  int n_players = 2;
  int n_chips = 10'000;
  int ante = 0;
};

class PokerState {
public:
  PokerState(int n_players = 2, int chips = 10'000, int ante = 0);
  PokerState(const PokerConfig& config);
  PokerState(const PokerState&) = default;
  PokerState(PokerState&&) = default;

  PokerState& operator=(const PokerState&) = default;
  PokerState& operator=(PokerState&&) = default;
  bool operator==(const PokerState& other) const = default;

  inline const std::vector<Player>& get_players() const { return _players; }
  inline const ActionHistory& get_action_history() const { return _actions; }
  inline int get_pot() const { return _pot; }
  inline int get_max_bet() const { return _max_bet; }
  inline uint8_t get_active() const { return _active; }
  inline uint8_t get_round() const { return _round; }
  inline uint8_t get_bet_level() const { return _bet_level; }
  inline int8_t get_winner() const { return _winner; }
  inline const std::vector<Action> get_biases() const { return _biases; }
  inline bool is_terminal() const { return get_winner() != -1 || get_round() >= 4; };
  int active_players() const;
  int vpip_players() const;
  PokerState apply(Action action) const;
  PokerState apply(const ActionHistory& action_history) const;
  PokerState apply_biases(const std::vector<Action>& biases) const;
  bool has_biases() const { return _biases.size() > _active && _biases[_active] != Action::BIAS_DUMMY; }
  std::string to_string() const;
  
  template <class Archive>
  void serialize(Archive& ar) {
    ar(_players, _biases, _actions, _pot, _max_bet, _active, _round, _bet_level, _winner);
  }

  uint8_t _first_bias = 10; // TODO: remove, just for asserts

private:
  std::vector<Player> _players;
  std::vector<Action> _biases;
  ActionHistory _actions;
  int _pot;
  int _max_bet;
  uint8_t _active;
  uint8_t _round;
  uint8_t _bet_level;
  int8_t _winner;

  PokerState bet(int amount) const;
  PokerState call() const;
  PokerState check() const;
  PokerState fold() const;
  PokerState bias(Action bias) const;
  PokerState next_state(Action action) const;
  void next_player();
  void next_round();
  void next_bias();
};

int total_bet_size(const PokerState& state, Action action);
std::vector<Action> valid_actions(const PokerState& state, const ActionProfile& profile);
int round_of_last_action(const PokerState& state);
std::vector<uint8_t> winners(const PokerState& state, const std::vector<Hand>& hands, const Board board_cards, const omp::HandEvaluator& eval);
int showdown_payoff(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, const omp::HandEvaluator& eval);
void deal_hands(Deck& deck, std::vector<std::array<uint8_t, 2>>& hands);
void deal_board(Deck& deck, std::array<uint8_t, 5>& board);
}

namespace std {
  template <>
  struct hash<pluribus::Hand> {
    size_t operator()(const pluribus::Hand& hand) const noexcept {
      size_t hash_value = 0;
      for(const auto& card : hand.cards()) {
        hash_value = hash_value * 31 + static_cast<size_t>(card);
      }
      return hash_value;
    }
  };
}
