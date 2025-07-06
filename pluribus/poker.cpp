#include <iostream>
#include <iomanip>
#include <cassert>
#include <vector>
#include <array>
#include <string>
#include <sstream>
#include <omp/Hand.h>
#include <omp/HandEvaluator.h>
#include <boost/functional/hash.hpp>
#include <pluribus/rng.hpp>
#include <pluribus/debug.hpp>
#include <pluribus/util.hpp>
#include <pluribus/poker.hpp>

namespace pluribus {

int Deck::draw() {
  uint8_t card;
  do {
    card = _cards[_current++];
  } while(_dead_cards.contains(card));
  return card;
}

void Deck::reset() {
  for(uint8_t i = 0; i < _cards.size(); ++i) {
    _cards[i] = i;
  }
  _current = 0;
}

void Deck::shuffle() {
  std::ranges::shuffle(_cards, GlobalRNG::instance());
  _current = 0;
}

uint64_t card_mask(const std::vector<uint8_t>& cards) {
  uint64_t mask = 0L;
  for(const uint8_t c : cards) mask |= card_mask(c);
  return mask;
}

const Hand Hand::PLACEHOLDER = Hand{MAX_CARDS, MAX_CARDS};

bool collides(const uint8_t card, const Hand& hand) {
  return hand.cards()[0] == card || hand.cards()[1] == card;
}

bool collides(const uint8_t card, const Board& board) {
  return std::ranges::find(board.cards(), card) != board.cards().end();
}

bool collides(const uint8_t card, const std::vector<uint8_t>& cards) {
  return std::ranges::find(cards, card) != cards.end();
}

bool collides(const Hand& h1, const Hand& h2) {
  return h1.cards()[0] == h2.cards()[0] || h1.cards()[1] == h2.cards()[1] ||
         h1.cards()[0] == h2.cards()[1] || h1.cards()[1] == h2.cards()[0];
}

bool collides(const Hand& hand, const Board& board) {
  return std::ranges::find(board.cards(), hand.cards()[0]) != board.cards().end() ||
         std::ranges::find(board.cards(), hand.cards()[1]) != board.cards().end();
}

bool collides(const Hand& hand, const std::vector<uint8_t>& cards) {
  return std::ranges::find(cards, hand.cards()[0]) != cards.end() ||
         std::ranges::find(cards, hand.cards()[1]) != cards.end();
}

std::vector<uint8_t> collect_cards(const Board& board, const Hand& hand, const int round) {
  const int card_sum = 2 + n_board_cards(round);
  std::vector<uint8_t> cards(card_sum);
  std::ranges::copy(hand.cards(), cards.data());
  if(round > 0) std::copy(board.cards().begin(), board.cards().begin() + card_sum - 2, cards.data() + 2);
  return cards;
}

void Player::invest(const int amount) {
  assert(!has_folded() && "Attempted to invest but player already folded.");
  assert(get_chips() >= amount && "Attempted to invest more chips than available.");
  _chips -= amount;
  _betsize += amount;
}

void Player::post_ante(const int amount) {
  _chips -= amount;
}

void Player::next_round() {
  _betsize = 0;
}

void Player::fold() {
  _folded = true;
}

void Player::reset(const int chips) {
  _chips = chips;
  _betsize = 0;
}

PokerState::PokerState(const int n_players, const int chips, const int ante, const bool straddle)
    : _pot{150}, _max_bet{100}, _round{0}, _bet_level{1}, _winner{-1}, _straddle{straddle} {
  _players.reserve(n_players);
  for(int i = 0; i < n_players; ++i) {
    _players.push_back(Player{chips});
  }
  
  if(_players.size() > 2) {
    _players[0].invest(50);
    _players[1].invest(100);
    if(straddle) {
      _players[2].invest(200);
      _pot += 200;
      _max_bet = 200;
      _active = n_players > 3 ? 3 : 0;
    }
    else {
      _active = 2;
    }
  }
  else {
    _players[0].invest(100);
    _players[1].invest(50);
    _active = 1;
  }

  if(ante > 0) {
    for(Player& p : _players) {
      p.post_ante(ante);
    }
    _pot += _players.size() * ante;
  }
}

PokerState::PokerState(const PokerConfig& config) : PokerState{config.n_players, config.n_chips, config.ante} {}

PokerState PokerState::next_state(const Action action) const {
  const Player& player = get_players()[get_active()];
  if(action == Action::ALL_IN) return bet(player.get_chips());
  if(action == Action::FOLD) return fold();
  if(action == Action::CHECK_CALL) return player.get_betsize() == _max_bet ? check() : call();
  if(is_bias(action)) return bias(action);
  return bet(total_bet_size(*this, action) - player.get_betsize());
}

int PokerState::active_players() const {
  int n = 0;
  for(const auto& p : _players) {
    if(!p.has_folded()) ++n;
  }
  return n;
}

PokerState PokerState::apply(const Action action) const {
  PokerState state = next_state(action);
  state._actions.push_back(action);
  return state;
}

PokerState PokerState::apply(const ActionHistory& action_history) const {
  PokerState state = *this;
  for(int i = 0; i < action_history.size(); ++i) {
    state = state.apply(action_history.get(i));
  }
  return state;
}

PokerState PokerState::apply_biases(const std::vector<Action>& biases) const {
  if(biases.size() != _players.size()) throw std::runtime_error("Number of biases to apply does not match number of players.");
  PokerState state = *this;
  state._biases = biases;
  return state;
}

std::string PokerState::to_string() const {
  std::ostringstream oss;
  oss << "============== " << round_to_str(_round) << ": " << std::setprecision(2) << std::fixed << _pot / 100.0 << " bb ==============\n";
  for(int i = 0; i < _players.size(); ++i) {
    oss << "Player " << i << "(" << _players[i].get_chips() / 100.0 << " bb): ";
    if(i == _active) {
      oss << "Active\n";
    }
    else if(_players[i].has_folded()) {
      oss << "Folded\n";
    }
    else {
      oss << _players[i].get_betsize() / 100.0 << " bb\n";
    }
  }
  return oss.str();
}

int8_t find_winner(const PokerState& state) {
  int8_t winner = -1;
  const std::vector<Player>& players = state.get_players();
  for(int8_t i = 0; i < players.size(); ++i) {
    if(!players[i].has_folded()) {
      if(winner == -1) winner = i;
      else return -1;
    }
  }
  return winner;
}

int big_blind_idx(const PokerState& state) {
  if(state.get_players().size() == 2) return 0;
  return state.is_straddle() ? 2 : 1;
}

int big_blind_size(const PokerState& state) {
  return state.is_straddle() ? 200 : 100;
}

std::string PokerConfig::to_string() const {
  std::ostringstream oss;
  oss << "PokerConfig{n_players=" << n_players << ", n_chips=" << n_chips << ", ante=" << ante;
  return oss.str();
}

PokerState PokerState::bet(const int amount) const {
  if(verbose) std::cout << std::fixed << std::setprecision(2) << "Player " << static_cast<int>(_active) << " (" 
                        << (_players[_active].get_chips() / 100.0) << "): " << (_bet_level == 0 ? "Bet " : "Raise to ")
                        << ((amount + _players[_active].get_betsize()) / 100.0) << " bb\n";
  assert(!_players[_active].has_folded() && "Attempted to bet but player already folded.");
  assert(_players[_active].get_chips() >= amount && "Not enough chips to bet.");
  assert(amount + _players[_active].get_betsize() > _max_bet && 
         "Attempted to bet but the players new betsize does not exceed the existing maximum bet.");
  assert(_winner == -1 && find_winner(*this) == -1 && "Attempted to bet but there are no opponents left.");
  PokerState state = *this;
  state._players[_active].invest(amount);
  state._pot += amount;
  state._max_bet = state._players[_active].get_betsize();
  ++state._bet_level;
  state.next_player();
  return state;
}

PokerState PokerState::call() const {
  const int amount = _max_bet - _players[_active].get_betsize();
  if(verbose) std::cout << std::fixed << std::setprecision(2) << "Player " << static_cast<int>(_active) << " (" 
                        << (_players[_active].get_chips() / 100.0) << "): Call " << (amount / 100.0) << " bb\n";
  assert(!_players[_active].has_folded() && "Attempted to call but player already folded.");
  assert(_max_bet > 0 && "Attempted call but no bet exists.");
  assert(_max_bet > _players[_active].get_betsize() && "Attempted call but player has already placed the maximum bet.");
  assert(_players[_active].get_chips() >= amount && "Not enough chips to call.");
  assert(_winner == -1 && find_winner(*this) == -1 && "Attempted to call but there are no opponents left.");
  PokerState state = *this;
  state._players[_active].invest(amount);
  state._pot += amount;
  state.next_player();
  return state;
}

PokerState PokerState::check() const {
  if(verbose) std::cout << std::fixed << std::setprecision(2) << "Player " << static_cast<int>(_active) << " (" 
                        << (_players[_active].get_chips() / 100.0) << "): Check\n";
  assert(!_players[_active].has_folded() && "Attempted to check but player already folded.");
  assert(_players[_active].get_betsize() == _max_bet && "Attempted check but a unmatched bet exists.");
  assert(_max_bet == 0 || (_round == 0 && _active == big_blind_idx(*this)) && "Attempted to check but a bet exists");
  assert(_winner == -1 && find_winner(*this) == -1 && "Attempted to check but there are no opponents left.");
  PokerState state = *this;
  state.next_player();
  return state;
}

PokerState PokerState::fold() const {
  if(verbose) std::cout << std::fixed << std::setprecision(2) << "Player " << static_cast<int>(_active) << " (" 
                        << (_players[_active].get_chips() / 100.0) << "): Fold\n";
  assert(!_players[_active].has_folded() && "Attempted to fold but player already folded.");
  assert(_max_bet > 0 && "Attempted fold but no bet exists.");
  assert(_players[_active].get_betsize() < _max_bet && "Attempted to fold but player can check");
  assert(_winner == -1 && find_winner(*this) == -1 && "Attempted to fold but there are no opponents left.");
  PokerState state = *this;
  state._players[_active].fold();
  state._winner = find_winner(state);
  if(state._winner == -1) {
    state.next_player();
  }
  else if(verbose) {
    std::cout << "Only player " << static_cast<int>(state._winner) << " is remaining.\n";
  }
  return state;
}

PokerState PokerState::bias(const Action bias) const {
  PokerState state = *this;
  if(state._biases.size() == 0) {
    state._first_bias = state._active;
    state._biases.resize(state._players.size(), Action::BIAS_DUMMY);
  }
  state._biases[state.get_active()] = bias;
  state.next_bias();
  return state;
}

uint8_t increment(uint8_t i, const uint8_t max_val) {
  return ++i > max_val ? 0 : i;
}

void PokerState::next_round() {
  ++_round;
  if(verbose) if(verbose) std::cout << std::fixed << std::setprecision(2) << round_to_str(_round) << " (" << (_pot / 100.0) << " bb):\n";
  for(Player& p : _players) {
    p.next_round();
  }
  _active = 0;
  _max_bet = 0;
  _bet_level = 0;
  if(_round < 4 && (_players[_active].has_folded() || _players[_active].get_chips() == 0)) next_player();
}

bool is_round_complete(const PokerState& state) {
  return state.get_players()[state.get_active()].get_betsize() == state.get_max_bet() && 
         (state.get_max_bet() > 0 || state.get_active() == 0) &&
         (state.get_max_bet() > big_blind_size(state) || state.get_active() != big_blind_idx(state) || state.get_round() != 0); // preflop, big blind
}

int round_of_last_action(const PokerState& state) {
  return state.get_round() == 0 || state.get_max_bet() > 0 || state.get_active() != 0 ? state.get_round() : state.get_round() - 1;
}

void PokerState::next_player() {
  do {
    _active = increment(_active, _players.size() - 1);
    if(is_round_complete(*this)) {
      next_round();
      return;
    }
  } while(_players[_active].has_folded() || _players[_active].get_chips() == 0);
}

void PokerState::next_bias() {
  const uint8_t init_player_idx = _active;
  do {
    _active = increment(_active, _players.size() - 1);
  } while(_active != init_player_idx && (_players[_active].has_folded() || _biases[_active] != Action::BIAS_DUMMY));
}

int total_bet_size(const PokerState& state, const Action action) {
  const Player& active_player = state.get_players()[state.get_active()];
  if(action == Action::ALL_IN) {
    return active_player.get_chips() + active_player.get_betsize();
  }
  if(action.get_bet_type() > 0.0f) {
    const int missing = state.get_max_bet() - active_player.get_betsize();
    const int real_pot = state.get_pot() + missing;
    return real_pot * action.get_bet_type() + missing + active_player.get_betsize();
  }
  throw std::runtime_error("Invalid action bet size: " + std::to_string(action.get_bet_type()));
}

std::vector<Action> valid_actions(const PokerState& state, const ActionProfile& profile) {
  const std::vector<Action>& actions = profile.get_actions(state.get_round(), state.get_bet_level(), state.get_active(), state.get_pot());
  std::vector<Action> valid;
  valid.reserve(actions.size());
  const Player& player = state.get_players()[state.get_active()];
  for(Action a : actions) {
    if(a == Action::CHECK_CALL) {
      valid.push_back(a);
      continue;
    }
    if(a == Action::FOLD) {
      if(player.get_betsize() < state.get_max_bet()) {
        valid.push_back(a);
      }
      continue;
    }
    const int total_bet = total_bet_size(state, a);
    if(const int required = total_bet - player.get_betsize(); required <= player.get_chips() && total_bet > state.get_max_bet()) {
      valid.push_back(a);
    }
  }
  return valid;
}

std::vector<uint8_t> winners(const PokerState& state, const std::vector<Hand>& hands, const Board& board_cards, const omp::HandEvaluator& eval) {
  int best = -1;
  std::vector<uint8_t> winners{};
  omp::Hand board = omp::Hand::empty();
  for(const uint8_t& idx : board_cards.cards()) {
    board += omp::Hand(idx);
  }
  for(uint8_t i = 0; i < hands.size(); ++i) {
    if(state.get_players()[i].has_folded()) continue;
    if(const uint16_t value = eval.evaluate(board + hands[i].cards()[0] + hands[i].cards()[1]); value == best) {
      winners.push_back(i);
    }
    else if(value > best) {
      best = value;
      winners.clear();
      winners.push_back(i);
    }
  }
  return winners;
}

int showdown_payoff(const PokerState& state, const int i, const Board& board, const std::vector<Hand>& hands, const omp::HandEvaluator& eval) {
  if(state.get_players()[i].has_folded()) return 0;
  std::vector<uint8_t> win_idxs = winners(state, hands, board, eval);
  return std::ranges::find(win_idxs, i) != win_idxs.end() ? state.get_pot() / win_idxs.size() : 0;
}

void deal_hands(Deck& deck, std::vector<std::array<uint8_t, 2>>& hands) {
  for(auto& hand : hands) {
    hand[0] = deck.draw();
    hand[1] = deck.draw();
  }
}

void deal_board(Deck& deck, std::array<uint8_t, 5>& board) {
  for(int i = 0; i < 5; ++i) board[i] = deck.draw();
}

}