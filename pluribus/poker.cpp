#include <array>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <boost/functional/hash.hpp>
#include <omp/Hand.h>
#include <omp/HandEvaluator.h>
#include <pluribus/debug.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/rng.hpp>
#include <pluribus/util.hpp>

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

uint64_t card_mask(const uint8_t* cards, const size_t n_cards) {
  uint64_t mask = 0L;
  for(size_t i = 0; i < n_cards; ++i) mask |= card_mask(cards[i]);
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

int blind_size(const SlimPokerState& state, const int pos) {
  switch(pos) {
    case 0: return state.get_players().size() > 2 ? 50 : 100;
    case 1: return state.get_players().size() > 2 ? 100 : 50;
    case 2: return state.is_straddle() ? 200 : 0;
    default: return 0;
  }
}

int big_blind_idx(const SlimPokerState& state) {
  if(state.get_players().size() == 2) return 0;
  return state.is_straddle() ? 2 : 1;
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

SlimPokerState::SlimPokerState(const int n_players, const std::vector<int>& chips, const int ante, const bool straddle)
    : _pot{150}, _max_bet{100}, _round{0}, _bet_level{1}, _winner{-1}, _straddle{straddle} {
  if(n_players != chips.size()) {
    throw std::runtime_error("Player amount mismatch: n_players=" + std::to_string(n_players) + ", chip stacks=" + std::to_string(chips.size()));
  }

  _players.reserve(n_players);
  for(int i = 0; i < n_players; ++i) {
    _players.push_back(Player{chips[i]});
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

SlimPokerState::SlimPokerState(const int n_players, const int chips, const int ante, const bool straddle)
    : SlimPokerState{n_players, std::vector(n_players, chips), ante, straddle} {}

SlimPokerState::SlimPokerState(const PokerConfig& config, int n_chips) : SlimPokerState{config.n_players, n_chips, config.ante, config.straddle} {}

bool SlimPokerState::has_player_vpip(const int pos) const {
  if(get_round() != 0) throw std::runtime_error("VPIP is only defined preflop.");
  return get_players()[pos].get_betsize() > blind_size(*this, pos);
}

bool SlimPokerState::is_in_position(const int pos) const {
  if(get_round() == 0) {
    for(int i = pos + 1; i < get_players().size(); ++i) {
      if(has_player_vpip(i)) return false;
    }
  }
  else {
    for(int i = pos + 1; i < get_players().size(); ++i) {
      if(!get_players()[i].has_folded()) return false;
    }
  }
  return true;
}

int SlimPokerState::active_players() const {
  int n = 0;
  for(const auto& p : _players) {
    if(!p.has_folded()) ++n;
  }
  return n;
}

void SlimPokerState::apply_in_place(const Action action) {
  const Player& player = get_players()[get_active()];
  if(action == Action::ALL_IN) return bet(player.get_chips());
  if(action == Action::FOLD) return fold();
  if(action == Action::CHECK_CALL) return player.get_betsize() == _max_bet ? check() : call();
  if(is_bias(action)) return bias(action);
  return bet(total_bet_size(*this, action) - player.get_betsize());
}

void SlimPokerState::apply_in_place(const ActionHistory& action_history) {
  for(int i = 0; i < action_history.size(); ++i) {
    apply_in_place(action_history.get(i));
  }
}

void SlimPokerState::apply_biases_in_place(const std::vector<Action>& biases) {
  if(biases.size() != get_players().size()) throw std::runtime_error("Number of biases to apply does not match number of players.");
  _biases = biases;
}

SlimPokerState SlimPokerState::apply_copy(const Action action) const {
  SlimPokerState state = *this;
  state.apply_in_place(action);
  return state;
}

SlimPokerState SlimPokerState::apply_copy(const ActionHistory& action_history) const{
  SlimPokerState state = *this;
  state.apply_in_place(action_history);
  return state;
}

SlimPokerState SlimPokerState::apply_biases_copy(const std::vector<Action>& biases) const {
  SlimPokerState state = *this;
  state.apply_biases_in_place(biases);
  return state;
}

int SlimPokerState::vpip_players() const {
  int n = 0;
  for(int i = 0; i < get_players().size(); ++i) {
    if(has_player_vpip(i)) ++n;
  }
  return n;
}

std::string SlimPokerState::to_string() const {
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

int8_t find_winner(const SlimPokerState& state) {
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

int big_blind_size(const SlimPokerState& state) {
  return state.is_straddle() ? 200 : 100;
}

std::string PokerConfig::to_string() const {
  std::ostringstream oss;
  oss << "PokerConfig{n_players=" << n_players << ", ante=" << ante << ", straddle=" << (straddle ? "true" : "false");
  return oss.str();
}

void SlimPokerState::bet(const int amount) {
  if(verbose) std::cout << std::fixed << std::setprecision(2) << "Player " << static_cast<int>(_active) << " (" 
                        << (_players[_active].get_chips() / 100.0) << "): " << (_bet_level == 0 ? "Bet " : "Raise to ")
                        << ((amount + _players[_active].get_betsize()) / 100.0) << " bb\n";
  assert(!_players[_active].has_folded() && "Attempted to bet but player already folded.");
  assert(_players[_active].get_chips() >= amount && "Not enough chips to bet.");
  assert(amount + _players[_active].get_betsize() > _max_bet && 
         "Attempted to bet but the players new betsize does not exceed the existing maximum bet.");
  assert(_winner == -1 && find_winner(*this) == -1 && "Attempted to bet but there are no opponents left.");
  _players[_active].invest(amount);
  _pot += amount;
  _max_bet = _players[_active].get_betsize();
  ++_bet_level;
  next_player();
}

void SlimPokerState::call() {
  const int amount = _max_bet - _players[_active].get_betsize();
  if(verbose) std::cout << std::fixed << std::setprecision(2) << "Player " << static_cast<int>(_active) << " (" 
                        << (_players[_active].get_chips() / 100.0) << "): Call " << (amount / 100.0) << " bb\n";
  assert(!_players[_active].has_folded() && "Attempted to call but player already folded.");
  assert(_max_bet > 0 && "Attempted call but no bet exists.");
  assert(_max_bet > _players[_active].get_betsize() && "Attempted call but player has already placed the maximum bet.");
  assert(_players[_active].get_chips() >= amount && "Not enough chips to call.");
  assert(_winner == -1 && find_winner(*this) == -1 && "Attempted to call but there are no opponents left.");
  _players[_active].invest(amount);
  _pot += amount;
  next_player();
}

void SlimPokerState::check() {
  if(verbose) std::cout << std::fixed << std::setprecision(2) << "Player " << static_cast<int>(_active) << " (" 
                        << (_players[_active].get_chips() / 100.0) << "): Check\n";
  assert(!_players[_active].has_folded() && "Attempted to check but player already folded.");
  assert(_players[_active].get_betsize() == _max_bet && "Attempted check but a unmatched bet exists.");
  assert(_max_bet == 0 || (_round == 0 && _active == big_blind_idx(*this)) && "Attempted to check but a bet exists");
  assert(_winner == -1 && find_winner(*this) == -1 && "Attempted to check but there are no opponents left.");
  next_player();
}

void SlimPokerState::fold() {
  if(verbose) std::cout << std::fixed << std::setprecision(2) << "Player " << static_cast<int>(_active) << " (" 
                        << (_players[_active].get_chips() / 100.0) << "): Fold\n";
  assert(!_players[_active].has_folded() && "Attempted to fold but player already folded.");
  assert(_max_bet > 0 && "Attempted fold but no bet exists.");
  assert(_players[_active].get_betsize() < _max_bet && "Attempted to fold but player can check");
  assert(_winner == -1 && find_winner(*this) == -1 && "Attempted to fold but there are no opponents left.");
  _players[_active].fold();
  _winner = find_winner(*this);
  if(_winner == -1) {
    next_player();
  }
  else if(verbose) {
    std::cout << "Only player " << static_cast<int>(_winner) << " is remaining.\n";
  }
}

void SlimPokerState::bias(const Action bias) {
  if(_biases.size() == 0) {
    _first_bias = _active;
    _biases.resize(_players.size(), Action::BIAS_DUMMY);
  }
  if(_biases[get_active()] != Action::BIAS_DUMMY) { // TODO: remove
    throw std::runtime_error("Player " + std::to_string(_active) + " already has a bias! Bias=" + _biases[get_active()].to_string());
  }
  _biases[get_active()] = bias;
  next_bias();
}

uint8_t increment(uint8_t i, const uint8_t max_val) {
  return ++i > max_val ? 0 : i;
}

void SlimPokerState::next_round() {
  ++_round;
  if(verbose) std::cout << std::fixed << std::setprecision(2) << round_to_str(_round) << " (" << (_pot / 100.0) << " bb):\n";
  for(Player& p : _players) {
    p.next_round();
  }
  _active = 0;
  _max_bet = 0;
  _bet_level = 0;
  if(_round < 4 && (_players[_active].has_folded() || _players[_active].get_chips() == 0)) next_player();
}

bool is_round_complete(const SlimPokerState& state) {
  return state.get_players()[state.get_active()].get_betsize() == state.get_max_bet() && 
         (state.get_max_bet() > 0 || state.get_active() == 0) &&
         (state.get_max_bet() > big_blind_size(state) || state.get_active() != big_blind_idx(state) || state.get_round() != 0); // preflop, big blind
}

int round_of_last_action(const SlimPokerState& state) {
  return state.get_round() == 0 || state.get_max_bet() > 0 || state.get_active() != 0 ? state.get_round() : state.get_round() - 1;
}

void SlimPokerState::next_player() {
  do {
    _active = increment(_active, _players.size() - 1);
    if(is_round_complete(*this)) {
      next_round();
      return;
    }
  } while(_players[_active].has_folded() || _players[_active].get_chips() == 0);
}

void SlimPokerState::next_bias() {
  const uint8_t init_player_idx = _active;
  do {
    _active = increment(_active, _players.size() - 1);
  } while(_active != init_player_idx && _players[_active].has_folded());
}

PokerState PokerState::apply(const Action action) const {
  PokerState state = *this;
  state.apply_in_place(action);
  if(!is_bias(action)) state._actions.push_back(action);
  return state;
}

PokerState PokerState::apply(const ActionHistory& action_history) const {
  if(action_history.get_history().empty()) return *this;
  PokerState state = apply(action_history.get(0));
  for(int i = 1; i < action_history.size(); ++i) {
    state = state.apply(action_history.get(i));
  }
  return state;
}

PokerState PokerState::apply_biases(const std::vector<Action>& biases) const {
  PokerState state = *this;
  state.apply_biases_in_place(biases);
  return state;
}

int total_bet_size(const SlimPokerState& state, const Action action) {
  const Player& active_player = state.get_players()[state.get_active()];
  if(action == Action::ALL_IN) {
    int max_total = 0;
    for(int i = 0; i < state.get_players().size(); ++i) {
      if(i != state.get_active() && !state.get_players()[i].has_folded()) {
        max_total = std::max(state.get_players()[i].get_betsize() + state.get_players()[i].get_chips(), max_total);
      }
    }
    return std::min(active_player.get_chips() + active_player.get_betsize(), max_total);
  }
  if(action.get_bet_type() > 0.0f) {
    const int missing = state.get_max_bet() - active_player.get_betsize();
    const int real_pot = state.get_pot() + missing;
    return real_pot * action.get_bet_type() + missing + active_player.get_betsize();
  }
  throw std::runtime_error("Invalid action bet size: " + std::to_string(action.get_bet_type()));
}

double fractional_bet_size(const SlimPokerState& state, const int total_size) {
  const double raise_size = total_size - state.get_max_bet();
  const double pot_size = state.get_pot() + state.get_max_bet() - state.get_players()[state.get_active()].get_betsize();
  return raise_size / pot_size;
}

std::vector<Action> valid_actions(const SlimPokerState& state, const ActionProfile& profile) {
  const std::vector<Action>& actions = profile.get_actions(state);
  std::vector<Action> valid;
  valid.reserve(actions.size());
  const Player& player = state.get_players()[state.get_active()];
  for(Action a : actions) {
    if(a == Action::CHECK_CALL) {
      valid.push_back(a);
      continue;
    }
    if(a == Action::FOLD) {
      if(player.get_betsize() < state.get_max_bet() && player.get_chips() > 0) {
        valid.push_back(a);
      }
      continue;
    }
    if(a == Action::ALL_IN) { // faster than calling total_bet_size
      if(player.get_chips() > state.get_max_bet() - player.get_betsize()) {
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

std::vector<uint8_t> winners(const SlimPokerState& state, const std::vector<Hand>& hands, const Board& board_cards, const omp::HandEvaluator& eval) {
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

int showdown_payoff(const SlimPokerState& state, const int i, const Board& board, const std::vector<Hand>& hands, const RakeStructure& rake,
    const omp::HandEvaluator& eval) {
  if(state.get_players()[i].has_folded()) return 0;
  std::vector<uint8_t> win_idxs = winners(state, hands, board, eval);
  return std::ranges::find(win_idxs, i) != win_idxs.end() ? rake.payoff(state) / win_idxs.size() : 0;
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