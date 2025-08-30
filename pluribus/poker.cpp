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
  for(int i = 0; i < _cards.size(); ++i) {
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

void Player::take_back(const int amount) {
  assert(amount <= _betsize && "Attempted to take back more chips than the player has invested.");
  _chips += amount;
  _betsize -= amount;
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

void Pot::add_side_pot(int amount, const std::vector<int>& player_idxs, const std::vector<Player>& players) {
  if(!_pots) _pots = new std::vector<SidePot>{};
  for(auto& [pot_amount, pot_players] : *_pots) {
    bool match = true;
    for(int p : pot_players) {
      if(!players[p].has_folded() && std::ranges::find(player_idxs, p) == player_idxs.end()) { // TODO: performance: && players[p].chips == 0
        match = false;
        break;
      }
    }
    if(match) {
      pot_amount += amount;
      return;
    }
  }
  _pots->emplace_back(amount, player_idxs);
}

SlimPokerState::SlimPokerState(const int n_players, const std::vector<int>& chips, const int ante, const bool straddle)
    : _pot{150}, _max_bet{100}, _min_raise{100}, _round{0}, _no_chips{0}, _bet_level{1}, _winner{-1}, _straddle{straddle} {
  if(n_players != chips.size()) {
    throw std::runtime_error("Player amount mismatch: n_players=" + std::to_string(n_players) + ", chip stacks=" + std::to_string(chips.size()));
  }

  _players.reserve(n_players);
  for(int i = 0; i < n_players; ++i) {
    _players.emplace_back(chips[i]);
  }

  if(_players.size() > 2) {
    _players[0].invest(50);
    _players[1].invest(100);
    if(straddle) {
      _players[2].invest(200);
      _pot.add(200);
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
    _pot.add(static_cast<int>(_players.size()) * ante);
  }
}

SlimPokerState::SlimPokerState(const int n_players, const int chips, const int ante, const bool straddle)
    : SlimPokerState{n_players, std::vector(n_players, chips), ante, straddle} {}

SlimPokerState::SlimPokerState(const PokerConfig& config, const int n_chips) : SlimPokerState{config.n_players, n_chips, config.ante, config.straddle} {}

bool SlimPokerState::has_player_vpip(const int pos) const {
  return get_players()[pos].get_betsize() > (get_round() == 0 ? blind_size(*this, pos) : 0);
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
  oss << "============== " << round_to_str(_round) << ": " << std::fixed << std::setprecision(2) << _pot.total() / 100.0 << " bb ==============\n";
  if(_pot.has_side_pots()) {
    const auto& pots = *_pot.get_side_pots();
    for(int i = 0; i < pots.size(); ++i) {
      oss << "Pot " << i << ": " << pots[i].amount / 100.0 << " bb (Players: ";
      for(int p = 0; p < pots[i].players.size(); ++p) oss << p << (p != pots[i].players.size() - 1 ? ", " : ")\n");
    }
  }
  if(!_biases.empty()) {
    oss << "Biases: " << actions_to_str(_biases) << "\n";
  }
  oss << "Bet level: " << static_cast<int>(_bet_level) << ", Max bet: " << _max_bet / 100.0 << " bb, Min raise: " << _min_raise / 100.0 << " bb\n";
  if(_winner != -1) oss << "Winner: " << pos_to_str(_winner, _players.size(), _straddle) << "\n";
  for(int i = 0; i < _players.size(); ++i) {
    oss << pos_to_str(i, _players.size(), _straddle) << " (" << _players[i].get_chips() / 100.0 << " bb): " << _players[i].get_betsize() / 100.0 << " bb";
    if(i == _active) {
      oss << " (active)";
    }
    else if(_players[i].has_folded()) {
      oss << "(folded)";
    }
    if(i != _players.size() - 1) oss << "\n";
  }
  return oss.str();
}

int8_t find_winner(const SlimPokerState& state) {
  int8_t winner = -1;
  const std::vector<Player>& players = state.get_players();
  for(int8_t i = 0; i < static_cast<int8_t>(players.size()); ++i) {
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
  auto& player = _players[_active];
  if(verbose) std::cout << std::fixed << std::setprecision(2) << "Player " << static_cast<int>(_active) << " (" 
                        << (player.get_chips() / 100.0) << "): " << (_bet_level == 0 ? "Bet " : "Raise to ")
                        << ((amount + player.get_betsize()) / 100.0) << " bb\n";
  assert(!player.has_folded() && "Attempted to bet but player already folded.");
  assert(player.get_chips() >= amount && "Not enough chips to bet.");
  assert(amount + player.get_betsize() - _max_bet >= _min_raise &&
         "Attempted to bet/raise but the players new betsize does not exceed the minimum bet/raise size.");
  assert(_winner == -1 && find_winner(*this) == -1 && "Attempted to bet but there are no opponents left.");

  player.invest(amount);
  _pot.add(amount);
  _min_raise = player.get_betsize() - _max_bet;
  _max_bet = player.get_betsize();
  ++_bet_level;
  _no_chips += player.get_chips() == 0;
  next_player();
}

void SlimPokerState::call() {
  auto& player = _players[_active];
  const int amount = std::min(_max_bet - player.get_betsize(), player.get_chips());
  if(verbose) std::cout << std::fixed << std::setprecision(2) << "Player " << static_cast<int>(_active) << " (" 
                        << (player.get_chips() / 100.0) << "): Call " << (amount / 100.0) << " bb\n";
  assert(!player.has_folded() && "Attempted to call but player already folded.");
  assert(_max_bet > 0 && "Attempted call but no bet exists.");
  assert(_max_bet > player.get_betsize() && "Attempted call but player has already placed the maximum bet.");
  assert(player.get_chips() >= amount && "Not enough chips to call.");
  assert(_winner == -1 && find_winner(*this) == -1 && "Attempted to call but there are no opponents left.");
  player.invest(amount);
  _pot.add(amount);
  _no_chips += player.get_chips() == 0;
  next_player();
}

void SlimPokerState::check() {
  const auto& player = _players[_active];
  if(verbose) std::cout << std::fixed << std::setprecision(2) << "Player " << static_cast<int>(_active) << " (" 
                        << (player.get_chips() / 100.0) << "): Check\n";
  assert(!player.has_folded() && "Attempted to check but player already folded.");
  assert(player.get_betsize() == _max_bet && "Attempted check but a unmatched bet exists.");
  assert(_max_bet == 0 || (_round == 0 && _active == big_blind_idx(*this)) && "Attempted to check but a bet exists");
  assert(_winner == -1 && find_winner(*this) == -1 && "Attempted to check but there are no opponents left.");
  next_player();
}

void SlimPokerState::fold() {
  auto& player = _players[_active];
  if(verbose) std::cout << std::fixed << std::setprecision(2) << "Player " << static_cast<int>(_active) << " (" 
                        << (player.get_chips() / 100.0) << "): Fold\n";
  assert(!player.has_folded() && "Attempted to fold but player already folded.");
  assert(_max_bet > 0 && "Attempted fold but no bet exists.");
  assert(player.get_betsize() < _max_bet && "Attempted to fold but player can check");
  assert(_winner == -1 && find_winner(*this) == -1 && "Attempted to fold but there are no opponents left.");
  assert(player.get_chips() > 0 && "Attempted to fold but player is all-in.");
  player.fold();
  _winner = find_winner(*this);
  ++_no_chips;
  if(_winner == -1) {
    next_player();
  }
  else if(verbose) {
    std::cout << "Only player " << static_cast<int>(_winner) << " is remaining.\n";
  }
}

void SlimPokerState::bias(const Action bias) {
  if(_biases.empty()) {
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
  if(verbose) std::cout << std::fixed << std::setprecision(2) << round_to_str(_round) << ":\n";
  if(_pot.has_side_pots()) {
    update_side_pots();
  }
  else {
    int expected_bet = -1;
    for(auto& p : _players) {
      if(!p.has_folded()) {
        if(expected_bet == -1) {
          expected_bet = p.get_betsize();
        }
        else if(expected_bet != p.get_betsize()) {
          init_side_pots();
          break;
        }
      }
    }
  }
  ++_round;
  for(Player& p : _players) p.next_round();
  _active = 0;
  _max_bet = 0;
  _min_raise = 100;
  _bet_level = 0;
  if(_round < 4 && (_players[_active].has_folded() || _players[_active].get_chips() == 0 || n_players_with_chips() == 1)) {
    next_player();
  }
}

bool SlimPokerState::is_round_complete() const {
  const auto& player = get_players()[get_active()];
  const bool is_done = n_players_with_chips() == 1;
  return player.get_betsize() == get_max_bet() &&
         (get_max_bet() > 0 || get_active() == 0 || is_done) &&
         (get_max_bet() > big_blind_size(*this) || get_active() != big_blind_idx(*this) || get_round() != 0 || is_done); // preflop, big blind
}

int round_of_last_action(const SlimPokerState& state) {
  return state.get_round() == 0 || state.get_max_bet() > 0 || state.get_active() != 0 ? state.get_round() : state.get_round() - 1;
}

void SlimPokerState::next_player() {
  do {
    _active = increment(_active, _players.size() - 1);
    if(is_round_complete()) {
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

void SlimPokerState::init_side_pots() {
  std::vector<int> p_idxs;
  int prev_amount = _pot.total();
  for(int i = 0; i < _players.size(); ++i) {
    // add folded players with invested chips to avoid creating a separate side pot during update_side_pots()
    if(!_players[i].has_folded() || _players[i].get_betsize() > 0) p_idxs.push_back(i);
    prev_amount -= _players[i].get_betsize();
  }
  _pot.add_side_pot(prev_amount, p_idxs, _players);
  update_side_pots();
}

void SlimPokerState::update_side_pots() {
  std::vector<int> p_idxs;
  for(int i = 0; i < _players.size(); ++i) {
    if(!_players[i].has_folded() || _players[i].get_betsize() > 0) p_idxs.push_back(i);
  }
  while(!p_idxs.empty()) {
    int amount = std::numeric_limits<int>::max();
    for(int i = static_cast<int>(p_idxs.size()) - 1; i >= 0; --i) {
      const Player& player = _players[p_idxs[i]];
      if(player.get_betsize() == 0) {
        p_idxs.erase(p_idxs.begin() + i);
      }
      else {
        amount = std::min(player.get_betsize(), amount);
      }
    }
    if(!p_idxs.empty()) {
      _pot.add_side_pot(amount * static_cast<int>(p_idxs.size()), p_idxs, _players);
      for(const int p_idx : p_idxs) {
        _players[p_idx].set_betsize(_players[p_idx].get_betsize() - amount);
      }
    }
  }
}

PokerState PokerState::apply(const Action action) const {
  PokerState state = *this;
  state.apply_in_place(action);
  return state;
}

PokerState PokerState::apply(const ActionHistory& action_history) const {
  PokerState state = *this;
  state.apply_in_place(action_history);
  return state;

}

void PokerState::apply_in_place(const Action action) {
  SlimPokerState::apply_in_place(action);
  if(!is_bias(action)) _actions.push_back(action);
}

void PokerState::apply_in_place(const ActionHistory& action_history) {
  if(!action_history.get_history().empty()) {
    for(int i = 0; i < action_history.size(); ++i) {
      apply_in_place(action_history.get(i));
    }
  }
}

PokerState PokerState::apply_biases(const std::vector<Action>& biases) const {
  PokerState state = *this;
  state.apply_biases_in_place(biases);
  return state;
}

int total_bet_size(const SlimPokerState& state, const float frac) {
  const Player& active_player = state.get_players()[state.get_active()];
  if(frac <= 0.0f) throw std::runtime_error("Invalid action bet size: " + std::to_string(frac));
  const int missing = state.get_max_bet() - active_player.get_betsize();
  const int real_pot = state.get_pot().total() + missing;
  // return static_cast<int>(std::round(static_cast<float>(real_pot) * action.get_bet_type())) + missing + active_player.get_betsize();
  return real_pot * frac + missing + active_player.get_betsize();
}

int total_bet_size(const SlimPokerState& state, const Action action) {
  const Player& active_player = state.get_players()[state.get_active()];
  if(action == Action::ALL_IN) return active_player.get_chips() + active_player.get_betsize();
  return total_bet_size(state, action.get_bet_type());
}

double fractional_bet_size(const SlimPokerState& state, const int total_size) {
  const double raise_size = total_size - state.get_max_bet();
  const double pot_size = state.get_pot().total() + state.get_max_bet() - state.get_players()[state.get_active()].get_betsize();
  return raise_size / pot_size;
}

bool is_action_valid(const Action a, const SlimPokerState& state) {
  const Player& player = state.get_players()[state.get_active()];
  if(a == Action::CHECK_CALL) return true;
  if(a == Action::FOLD) return player.get_betsize() < state.get_max_bet() && player.get_chips() > 0;
  if(state.n_players_with_chips() == 1) return false;
  const int total_bet = total_bet_size(state, a);
  const int required = total_bet - player.get_betsize();
  // TODO: bets below the min_raise are allowed when no one has bet yet and the player's stack is less than the min_raise
  // TODO: what about when a previous player has bet all-in less than the min raise - can the next player raise less than the min raise if his stack is
  //       less than the min_raise? is the next min raise decreased due to the all-in raise?
  if(required <= player.get_chips() && total_bet - state.get_max_bet() >= state.get_min_raise()) {
    bool can_bet = false;
    for(int p_idx = 0; p_idx < state.get_players().size(); ++p_idx) {
      const Player& opponent = state.get_players()[p_idx];
      can_bet |= !opponent.has_folded() && p_idx != state.get_active() && opponent.get_betsize() + opponent.get_chips() > state.get_max_bet();
    }
    if(can_bet) return true;
  }
  return false;
}

std::vector<Action> valid_actions(const SlimPokerState& state, const ActionProfile& profile) {
  const std::vector<Action>& actions = profile.get_actions(state);
  std::vector<Action> valid;
  valid.reserve(actions.size());
  for(const Action a : actions) {
    if(is_action_valid(a, state)) valid.push_back(a);
  }
  return valid;
}

uint16_t score_hands(uint16_t scores[], const std::vector<Player>& players, const std::vector<Hand>& hands, const Board& board_cards,
    const omp::HandEvaluator& eval) {
  uint16_t best = 0;
  omp::Hand board = omp::Hand::empty();
  for(const uint8_t& idx : board_cards.cards()) {
    board += omp::Hand(idx);
  }
  for(int i = 0; i < hands.size(); ++i) {
    if(!players[i].has_folded()) {
      const uint16_t score = eval.evaluate(board + hands[i].cards()[0] + hands[i].cards()[1]);
      scores[i] = score;
      best = std::max(score, best);
    }
  }
  return best;
}

int side_pot_payoff(const SlimPokerState& state, const int i, const Board& board, const std::vector<Hand>& hands, const RakeStructure& rake,
    const omp::HandEvaluator& eval) {
  // TODO: collapse side pots to distribute odd chips correctly - pop all players that have folded from all pots, combine equal pots.
  // two odd chip pots -> one even chip pot, removes odd chip bias to the first winner
  uint16_t scores[MAX_PLAYERS]{};
  const auto best = score_hands(scores, state.get_players(), hands, board, eval);
  const int total_payoff = rake.payoff(state.get_round(), state.get_pot().total());
  int payoff = 0;
  for(const auto& [amount, pot_players] : *state.get_pot().get_side_pots()) {
    if(scores[i] < best) continue;
    int n_winners = 0;
    int first_winner = -1;
    bool found = false;
    for(const int p_idx : pot_players) {
      found |= p_idx == i;
      if(!state.get_players()[p_idx].has_folded() && scores[p_idx] == best) {
        ++n_winners;
        if(first_winner == -1) first_winner = p_idx;
      }
    }
    if(found) payoff += amount / n_winners + (first_winner == i ? amount % n_winners : 0);
  }
  return static_cast<int>(std::round(static_cast<float>(payoff) / static_cast<float>(state.get_pot().total()) * static_cast<float>(total_payoff)));
}

int no_side_pot_payoff(const SlimPokerState& state, const int i, const Board& board, const std::vector<Hand>& hands, const RakeStructure& rake,
    const omp::HandEvaluator& eval) {
  uint16_t scores[MAX_PLAYERS]{};
  const auto best = score_hands(scores, state.get_players(), hands, board, eval);
  bool winner = false;
  int n_winners = 0;
  int first_winner = -1;
  for(int p = 0; p < state.get_players().size(); ++p) {
    if(!state.get_players()[p].has_folded() && scores[p] == best) {
      winner |= p == i;
      ++n_winners;
      if(first_winner == -1) first_winner = p;
    }
  }
  const int payoff = rake.payoff(state.get_round(), state.get_pot().total());
  return winner ? payoff / n_winners + (first_winner == i ? payoff % n_winners : 0) : 0;
}

int showdown_payoff(const SlimPokerState& state, const int i, const Board& board, const std::vector<Hand>& hands, const RakeStructure& rake,
    const omp::HandEvaluator& eval) {
  if(state.get_players()[i].has_folded()) return 0;
  return state.get_pot().has_side_pots() ? side_pot_payoff(state, i, board, hands, rake, eval) : no_side_pot_payoff(state, i, board, hands, rake, eval);

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
