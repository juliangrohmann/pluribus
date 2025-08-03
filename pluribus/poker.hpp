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
  explicit Deck(const std::unordered_set<uint8_t>& dead_cards = {}) : _cards{}, _dead_cards{dead_cards} { reset(); }
  explicit Deck(const std::vector<uint8_t>& dead_cards) : _cards{} {
    std::ranges::copy(dead_cards, std::inserter(_dead_cards, _dead_cards.end()));
    reset();
  }

  int draw();
  void add_dead_card(const uint8_t card) { _dead_cards.insert(card); }

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

inline uint64_t card_mask(const uint8_t card) { return 1L << card; }
uint64_t card_mask(const std::vector<uint8_t>& cards);

template<int N>
class CardSet {
public:
  CardSet() : _mask{0} {
    _cards.fill(0); 
    update_mask();
  }
  CardSet(const std::initializer_list<uint8_t>& cards) : _mask{0} {
    std::copy(cards.begin(), cards.end(), _cards.begin());
    update_mask(0, cards.size());
  }
  explicit CardSet(const std::array<uint8_t, N>& cards) : _cards{cards}, _mask{0}  {
    update_mask();
  }
  explicit CardSet(const std::vector<uint8_t>& cards) : _mask{0} {
    std::copy(cards.begin(), cards.end(), _cards.begin()); 
    update_mask(0, cards.size());
  }
  explicit CardSet(const std::string& card_str) : _mask{0} {
    str_to_cards(card_str, _cards.data()); 
    update_mask(0, card_str.size() / 2);
  }
   explicit CardSet(Deck& deck, const std::vector<uint8_t>& init_cards = {}) : _mask{0} {
    deal(deck, init_cards); 
  }

  void set_card(int i, uint8_t card) {
    _cards[i] = card;
    update_mask(i, i + 1);
  }

  bool collides(const CardSet& other) const {
    return _mask & other._mask;
  }

  void deal(Deck& deck, std::vector<uint8_t> init_cards = {}) { 
    for(int i = 0; i < init_cards.size(); ++i) _cards[i] = init_cards[i];
    for(int i = static_cast<int>(init_cards.size()); i < N; ++i) _cards[i] = deck.draw();
    update_mask();
  }

  uint64_t mask() const { return _mask; }
  const std::array<uint8_t, N>& cards() const { return _cards; }
  std::vector<uint8_t> as_vector(int n = -1) const { 
    return std::vector<uint8_t>{_cards.data(), _cards.data() + (n == -1 ? _cards.size() : n)}; 
  }
  std::string to_string() const { return cards_to_str(_cards.data(), N); }

  bool operator==(const CardSet&) const = default;

protected:
  void update_mask(int i = 0, const int n = N) {
    _mask = 0L;
    for(; i < n; ++i) _mask |= card_mask(_cards[i]);
  }

  std::array<uint8_t, N> _cards;
  uint64_t _mask;
};

class Board : public CardSet<5> {
public:
  Board() = default;
  Board(const std::initializer_list<uint8_t>& cards) : CardSet{cards} {}
  explicit Board(const std::array<uint8_t, 5>& cards) : CardSet{cards} {}
  explicit Board(const std::vector<uint8_t>& cards) : CardSet{cards} {}
  explicit Board(const std::string& card_str) : CardSet{card_str} {}
  explicit Board(Deck& deck, const std::vector<uint8_t>& init_cards = {}) : CardSet{deck, init_cards} {}
  bool operator==(const Board&) const = default;
};

class Hand : public CardSet<2> {
public:
  Hand() = default;
  Hand(const std::initializer_list<uint8_t>& cards) : CardSet{cards} {}
  explicit Hand(const std::array<uint8_t, 2>& cards) : CardSet{cards} {}
  explicit Hand(const std::vector<uint8_t>& cards) : CardSet{cards} {}
  explicit Hand(const std::string& card_str) : CardSet{card_str} {}
  explicit Hand(Deck& deck, const std::vector<uint8_t>& init_cards = {}) : CardSet{deck, init_cards} {}
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
bool collides(uint8_t card, const std::vector<uint8_t>& cards);
bool collides(const Hand& h1, const Hand& h2);
bool collides(const Hand& hand, const Board& board);
bool collides(const Hand& hand, const std::vector<uint8_t>& cards);
std::vector<uint8_t> collect_cards(const Board& board, const Hand& hand, int round = 3);

int big_blind_idx(const PokerState& state);
int blind_size(const PokerState& state, int pos);

class Player {
public:
  explicit Player(const int chips = 10'000) : _chips{chips} {}
  Player(const Player&) = default;
  Player(Player&&) = default;

  Player& operator=(const Player&) = default;
  Player& operator=(Player&&) = default;
  bool operator==(const Player& other) const = default;

  int get_chips() const { return _chips; }
  int get_betsize() const { return _betsize; }
  bool has_folded() const { return _folded; }
  void invest(int amount);
  void post_ante(int amount);
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
    int n_chips; // buffer for compatibility
    ar(n_players, n_chips, ante, straddle);
  }

  int n_players = 2;
  int ante = 0;
  bool straddle = false;
};

class PokerState {
public:
  explicit PokerState(int n_players, const std::vector<int>& chips, int ante = 0, bool straddle = false);
  explicit PokerState(int n_players = 2, int chips = 10'000, int ante = 0, bool straddle = false);
  explicit PokerState(const PokerConfig& config, int n_chips);
  PokerState(const PokerState&) = default;
  PokerState(PokerState&&) = default;

  PokerState& operator=(const PokerState&) = default;
  PokerState& operator=(PokerState&&) = default;
  bool operator==(const PokerState& other) const = default;

  const std::vector<Player>& get_players() const { return _players; }
  const ActionHistory& get_action_history() const { return _actions; }
  bool is_straddle() const { return _straddle; }
  int get_pot() const { return _pot; }
  int get_max_bet() const { return _max_bet; }
  uint8_t get_active() const { return _active; }
  uint8_t get_round() const { return _round; }
  uint8_t get_bet_level() const { return _bet_level; }
  int8_t get_winner() const { return _winner; }
  const std::vector<Action>& get_biases() const { return _biases; }
  bool is_terminal() const { return get_winner() != -1 || get_round() >= 4; }
  bool has_player_vpip(int pos) const;
  bool is_in_position(int pos) const;
  int vpip_players() const;
  int active_players() const;
  [[nodiscard]] PokerState apply(Action action) const;
  [[nodiscard]] PokerState apply(const ActionHistory& action_history) const;
  [[nodiscard]] PokerState apply_biases(const std::vector<Action>& biases) const;
  bool has_biases() const { return _biases.size() > _active && _biases[_active] != Action::BIAS_DUMMY; }
  std::string to_string() const;
  
  template <class Archive>
  void serialize(Archive& ar) {
    ar(_players, _biases, _actions, _pot, _max_bet, _active, _round, _bet_level, _winner, _straddle);
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
  bool _straddle;

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

class RakeStructure {
public:
  RakeStructure(const double percent, const double cap) : _percent{percent}, _cap{cap} {}

  int payoff(const PokerState& state) const {
    return state.get_round() == 0 ? state.get_pot() : static_cast<int>(round(std::max(state.get_pot() * (1.0 - _percent), state.get_pot() - _cap)));
  }

  bool operator==(const RakeStructure& other) const = default;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_percent, _cap);
  }

private:
  double _percent;
  double _cap;
};

int total_bet_size(const PokerState& state, Action action);
double fractional_bet_size(const PokerState& state, int total_size);
std::vector<Action> valid_actions(const PokerState& state, const ActionProfile& profile);
int round_of_last_action(const PokerState& state);
std::vector<uint8_t> winners(const PokerState& state, const std::vector<Hand>& hands, const Board& board_cards, const omp::HandEvaluator& eval);
int showdown_payoff(const PokerState& state, int i, const Board& board, const std::vector<Hand>& hands, const RakeStructure& rake,
    const omp::HandEvaluator& eval);
void deal_hands(Deck& deck, std::vector<std::array<uint8_t, 2>>& hands);
void deal_board(Deck& deck, std::array<uint8_t, 5>& board);
}

template <>
struct std::hash<pluribus::Hand> {
  size_t operator()(const pluribus::Hand& hand) const noexcept {
    size_t hash_value = 0;
    for(const auto& card : hand.cards()) {
      hash_value = hash_value * 31 + static_cast<size_t>(card);
    }
    return hash_value;
  }
};
