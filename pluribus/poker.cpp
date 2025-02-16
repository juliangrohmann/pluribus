#include <iostream>
#include <iomanip>
#include <cassert>
#include <vector>
#include <array>
#include <initializer_list>
#include <omp/Hand.h>
#include <omp/HandEvaluator.h>
#include <boost/functional/hash.hpp>
#include <pluribus/debug.hpp>
#include <pluribus/poker.hpp>

namespace pluribus {

void Deck::reset() {
  for(uint8_t i = 0; i < _cards.size(); ++i) {
    _cards[i] = i;
  }
  _current = 0;
}

void Deck::shuffle() {
  std::shuffle(_cards.begin(), _cards.end(), _rng);
  _current = 0;
}

void Player::invest(int amount) {
  assert(!has_folded() && "Attempted to invest but player already folded.");
  assert(get_chips() >= amount && "Attempted to invest more chips than available.");
  _chips -= amount;
  _betsize += amount;
}

void Player::next_round() {
  _betsize = 0;
}

void Player::fold() {
  _folded = true;
}

void Player::reset(int chips) {
  _chips = chips;
  _betsize = 0;
}

PokerState::PokerState(int n_players, int chips, int ante) : _pot{150}, _max_bet{100}, _bet_level{1}, _action_counter{0}, _round{0}, _winner{-1} {
  _players.reserve(n_players);
  for(int i = 0; i < n_players; ++i) {
    _players.push_back(Player{chips});
  }

  if(_players.size() > 2) {
    _players[0].invest(50);
    
    _players[1].invest(100);
    _active = 2;
  }
  else {
    _players[0].invest(100);
    _players[1].invest(50);
    _active = 1;
  }

  if(ante > 0) {
    for(Player p : _players) {
      p.invest(ante);
    }
    _pot += _players.size() * ante;
  }
}

PokerState PokerState::next_state(Action action) const {
  const Player& player = get_players()[get_active()];
  switch(action) {
    case Action::FOLD: return fold(); break;
    case Action::CHECK_CALL: return player.get_betsize() == _max_bet ? check() : call(); break;
    case Action::ALL_IN: return bet(player.get_chips()); break;
    default: return bet(total_bet_size(*this, action) - player.get_betsize()); break;
  }
}

PokerState PokerState::apply(Action action) const {
  PokerState state = next_state(action);
  state._actions.push_back(action);
  return state;
}

PokerState PokerState::bet(int amount) const {
  if(verbose) std::cout << std::fixed << std::setprecision(2) << "Player " << static_cast<int>(_active) << " (" 
                        << (_players[_active].get_chips() / 100.0) << "): " << (_bet_level == 0 ? "Bet " : "Raise to ")
                        << ((amount + _players[_active].get_betsize()) / 100.0) << " bb\n";
  assert(!_players[_active].has_folded() && "Attempted to bet but player already folded.");
  assert(_players[_active].get_chips() >= amount && "Not enough chips to bet.");
  assert(amount + _players[_active].get_betsize() > _max_bet && 
         "Attempted to bet but the players new betsize does not exceed the existing maximum bet.");
  PokerState state = *this;
  state._players[_active].invest(amount);
  state._pot += amount;
  state._max_bet = state._players[_active].get_betsize();
  ++state._bet_level;
  state.next_player();
  return state;
}

PokerState PokerState::call() const {
  int amount = _max_bet - _players[_active].get_betsize();
  if(verbose) std::cout << std::fixed << std::setprecision(2) << "Player " << static_cast<int>(_active) << " (" 
                        << (_players[_active].get_chips() / 100.0) << "): Call " << (amount / 100.0) << " bb\n";
  assert(!_players[_active].has_folded() && "Attempted to call but player already folded.");
  assert(_max_bet > 0 && "Attempted call but no bet exists.");
  assert(_max_bet > _players[_active].get_betsize() && "Attempted call but player has already placed the maximum bet.");
  assert(_players[_active].get_chips() >= amount && "Not enough chips to call.");
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
  assert(_max_bet == 0 || (_round == 0 && _active == 1) && "Attempted to check but a bet exists");
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
  PokerState state = *this;
  state._players[_active].fold();
  state.next_player();
  return state;
}

uint8_t increment(uint8_t i, uint8_t max_val) {
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
  // std::cout << "Max bet = " << state.get_max_bet() << ", Betsize = " << state.get_players()[state.get_active()].get_betsize() << "\n";
  return state.get_players()[state.get_active()].get_betsize() == state.get_max_bet() && 
         (state.get_max_bet() > 0 || state.get_active() == 0) &&
         (state.get_max_bet() > 100 || state.get_active() != 1 || round != 0); // preflop, big blind
}

void PokerState::next_player() {
  uint8_t init_player_idx = _active;
  bool all_in = false;
  do {
    _active = increment(_active, _players.size() - 1);
    if(is_round_complete(*this)) {
      next_round();
      return;
    }
    if(_active == init_player_idx && !all_in) {
      assert(!_players[init_player_idx].has_folded() && "All players folded and there are no winners.");
      _winner = init_player_idx;
      return;
    }
    else {
      all_in |= _players[_active].get_chips() == 0 && !_players[_active].has_folded();
    }
  } while((_players[_active].has_folded() || _players[_active].get_chips() == 0));
}

int total_bet_size(const PokerState& state, Action action) {
  switch(action) {
    case Action::BET_33: return state.get_pot() * 0.33;
    case Action::BET_50: return state.get_pot() * 0.50;
    case Action::BET_75: return state.get_pot() * 0.75;
    case Action::BET_100: return state.get_pot() * 1.00;
    case Action::BET_150: return state.get_pot() * 1.50;
    case Action::PREFLOP_2_BET: return state.get_players()[state.get_active()].get_chips() > 5000 ? 300 : 250;
    case Action::PREFLOP_3_BET: return state.get_pot() * 2.00;
    case Action::PREFLOP_4_BET: return state.get_pot() * 1.66;
    case Action::PREFLOP_5_BET: return state.get_pot() * 1.5;
    case Action::POSTFLOP_2_BET: return state.get_pot() * 1.5;
    case Action::POSTFLOP_3_BET: return state.get_pot() * 1.5;
    case Action::ALL_IN: return state.get_players()[state.get_active()].get_chips() + state.get_players()[state.get_active()].get_betsize();
    default: 
    std::cout << "Action: " << static_cast<int>(action) << "\n";
    assert(false && "Unknown bet action.");
    return -1;
  }
}

std::vector<Action> all_actions(const PokerState& state) {
  if(state.get_round() == 0) {
    switch(state.get_bet_level()) {
      case 1: return {Action::FOLD, Action::CHECK_CALL, Action::PREFLOP_2_BET};
      case 2: return {Action::FOLD, Action::CHECK_CALL, Action::PREFLOP_3_BET};
      case 3: return {Action::FOLD, Action::CHECK_CALL, Action::PREFLOP_4_BET};
      case 4: return {Action::FOLD, Action::CHECK_CALL, Action::PREFLOP_5_BET};
      default: return {Action::FOLD, Action::CHECK_CALL};
    }
  }
  else {
    switch(state.get_bet_level()) {
      case 0: return state.get_round() == 1 ? std::vector<Action>{Action::CHECK_CALL, Action::BET_33, Action::BET_75} : 
                                              std::vector<Action>{Action::CHECK_CALL, Action::BET_50, Action::BET_100, Action::BET_150};
      case 1: return {Action::FOLD, Action::CHECK_CALL, Action::POSTFLOP_2_BET};
      case 2: return {Action::FOLD, Action::CHECK_CALL, Action::POSTFLOP_3_BET};
      default: return {Action::FOLD, Action::CHECK_CALL};
    }
  }
}

std::vector<Action> valid_actions(const PokerState& state) {
  std::vector<Action> actions = all_actions(state);
  bool removed = false;
  const Player& player = state.get_players()[state.get_active()];
  for(int i = actions.size() - 1; i >= 0; --i) {
    if(actions[i] == Action::CHECK_CALL || actions[i] == Action::FOLD) continue;
    int required = total_bet_size(state, actions[i]) - player.get_betsize();
    if(required > player.get_chips()) {
      removed = true;
      actions.erase(actions.begin() + i);
    }
  }
  if(removed && player.get_chips() > state.get_max_bet()) {
    actions.push_back(Action::ALL_IN);
  }
  return actions;
}

std::vector<uint8_t> winners(const PokerState& state, const std::vector<std::array<uint8_t, 2>>& hands, 
               const std::array<uint8_t, 5>& board_cards, const omp::HandEvaluator& eval) {
  int best = -1;
  std::vector<uint8_t> winners{};
  omp::Hand board = omp::Hand::empty();
  for(const uint8_t& idx : board_cards) {
    board += omp::Hand(idx);
  }
  for(uint8_t i = 0; i < hands.size(); ++i) {
    if(state.get_players()[i].has_folded()) continue;
    uint16_t value = eval.evaluate(board + hands[i][0] + hands[i][1]);
    if(value == best) {
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