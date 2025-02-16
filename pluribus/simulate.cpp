#include <iostream>
#include <iomanip>
#include <cassert>
#include <tqdm/tqdm.hpp>
#include <omp/HandEvaluator.h>
#include <pluribus/util.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/debug.hpp>
#include <pluribus/simulate.hpp>

namespace pluribus {

std::vector<long> get_payoffs(const std::array<uint8_t, 5>& board, const std::vector<std::array<uint8_t, 2>>& hands,
                                 const PokerState& state, const omp::HandEvaluator& eval) {
  std::vector<long> payoffs;
  payoffs.reserve(hands.size());
  if(state.get_round() <= 3) {
    if(verbose) std::cout << "Player " << static_cast<int>(state.get_winner()) << " wins without showdown.\n";
    for(int i = 0; i < hands.size(); ++i) {
      payoffs[i] = state.get_winner() == i ? state.get_pot() : 0;
    }
  }
  else {
    std::vector<uint8_t> win_idxs = winners(state, hands, board, eval);
    if(verbose) std::cout << "Player" << (win_idxs.size() > 1 ? "s " : " ");
    for(int i = 0; i < hands.size(); ++i) {
      bool is_winner = std::find(win_idxs.begin(), win_idxs.end(), i) != win_idxs.end();
      payoffs[i] = is_winner ? state.get_pot() / win_idxs.size(): 0;
      if(verbose && is_winner) std::cout << i << (i == win_idxs[win_idxs.size() - 1] ? " " : ", ");
    }
    if(verbose) std::cout << "win" << (win_idxs.size() > 1 ? " " : "s ") << "at showdown.\n";
    payoffs[win_idxs[0]] += state.get_pot() % win_idxs.size();
  }
  return payoffs;
}

std::vector<std::vector<long>> simulate(std::vector<Agent*> agents, int n_chips, int ante, unsigned long n_iter) {
  omp::HandEvaluator eval;
  Deck deck;
  std::array<uint8_t, 5> board;
  std::vector<std::array<uint8_t, 2>> hands{agents.size()};
  std::vector<std::vector<long>> results{hands.size()};
  for(unsigned long t : tq::trange(n_iter)) {
    PokerState state(hands.size(), n_chips, ante);
    deck.shuffle();
    deal_board(deck, board);
    deal_hands(deck, hands);
    if(verbose) {
      for(int i = 0; i < hands.size(); ++i) {
        std::cout << "Player " << i << ": " << cards_to_str(hands[i].data(), 2) << "\n";
      }
      std::cout << "Board: " << cards_to_str(board.data(), 5) << "\n";
    }

    while(state.get_round() <= 3 && state.get_winner() == -1) {
      state = state.apply(agents[state.get_active()]->act(state));
    }

    std::vector<long> payoffs = get_payoffs(board, hands, state, eval);
    int net_winnings = 0;
    for(int i = 0; i < hands.size(); ++i) {
      int winnings = state.get_players()[i].get_chips() - n_chips + payoffs[i];
      if(verbose) {
        std::cout << std::setprecision(2) << std::fixed << "Player: " << static_cast<int>(i) << ": " 
                  << std::setw(8) << std::showpos << (winnings / 100.0) << " bb\n" << std::noshowpos;
      }
      results[i].push_back(results[i].size() > 0 ? results[i][results[i].size() - 1] + winnings : winnings);
      net_winnings += winnings;
    }
    assert(net_winnings == 0 && "Winnings are not zero sum.");
  }
  return results;
}

std::vector<long> simulate_round(std::array<uint8_t, 5> board, std::vector<std::array<uint8_t, 2>> hands, ActionHistory actions,
           int n_chips, int ante) {
  omp::HandEvaluator eval;
  PokerState state(hands.size(), n_chips, ante);
  if(verbose) {
    for(int i = 0; i < hands.size(); ++i) {
      std::cout << "Player " << i << ": " << cards_to_str(hands[i].data(), 2) << "\n";
    }
    std::cout << "Board: " << cards_to_str(board.data(), 5) << "\n";
  }

  for(int i = 0; i < actions.size(); ++i) {
    state = state.apply(actions.get(i));
  }
  if(state.get_round() <= 3 && state.get_winner() == -1) {
    if(verbose) std::cout << "The round is unfinished.\n";
    return std::vector<long>(hands.size(), 0);
  }
  else {
    std::vector<long> results = get_payoffs(board, hands, state, eval);
    int net_winnings = 0;
    for(int i = 0; i < hands.size(); ++i) {
      results[i] += state.get_players()[i].get_chips() - n_chips;
      if(verbose) {
        std::cout << std::setprecision(2) << std::fixed << "Player: " << static_cast<int>(i) << ": " 
                  << std::setw(8) << std::showpos << (results[i] / 100.0) << " bb\n" << std::noshowpos;
      }
      net_winnings += results[i];
    }
    assert(net_winnings == 0 && "Winnings are not zero sum.");
    return results;
  }
}

}
