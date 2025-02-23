#include <iostream>
#include <iomanip>
#include <cassert>
#include <omp.h>
#include <tqdm/tqdm.hpp>
#include <omp/HandEvaluator.h>
#include <pluribus/util.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/debug.hpp>
#include <pluribus/simulate.hpp>

namespace pluribus {

std::vector<long> get_payoffs(const Board& board, const std::vector<Hand>& hands, const PokerState& state, const omp::HandEvaluator& eval) {
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

std::vector<long> simulate(std::vector<Agent*> agents, int n_chips, int ante, long n_iter) {
  std::vector<std::vector<long>> thread_results{agents.size()};
  int ntid = omp_get_max_threads();
  long log_interval = n_iter / ntid / 100;
  for(auto& result : thread_results) result.resize(ntid);

  #pragma omp parallel for schedule(static)
  for(long t = 0; t < n_iter; ++t) {
    thread_local omp::HandEvaluator eval;
    thread_local Deck deck;
    thread_local Board board;
    thread_local std::vector<Hand> hands{agents.size()};
    int tid = omp_get_thread_num();
    if(tid == 0 && t % log_interval == 0) std::cout << "Sim: " << std::setprecision(1) << std::fixed << t / static_cast<double>(log_interval) << "%\n";
    PokerState state(hands.size(), n_chips, ante);
    deck.shuffle();
    board.deal(deck);
    for(Hand& hand : hands) hand.deal(deck);
    if(verbose) {
      for(int i = 0; i < hands.size(); ++i) {
        std::cout << "Player " << i << ": " << cards_to_str(hands[i].cards().data(), 2) << "\n";
      }
      std::cout << "Board: " << cards_to_str(board.cards().data(), 5) << "\n";
    }

    while(state.get_round() <= 3 && state.get_winner() == -1) {
      state = state.apply(agents[state.get_active()]->act(state, board, hands[state.get_active()], agents.size(), n_chips, ante));
    }

    std::vector<long> payoffs = get_payoffs(board, hands, state, eval);
    long net_round = 0l;
    for(int i = 0; i < hands.size(); ++i) {
      int winnings = state.get_players()[i].get_chips() - n_chips + payoffs[i];
      if(verbose) {
        std::cout << std::setprecision(2) << std::fixed << "Player: " << static_cast<int>(i) << ": " 
                  << std::setw(8) << std::showpos << (winnings / 100.0) << " bb\n" << std::noshowpos;
      }
      thread_results[i][tid] += winnings;
      net_round += winnings;
    }
    assert(net_round == 0l && "Round winnings are not zero sum.");
  }

  long net_winnings = 0l;
  std::vector<long> results(static_cast<int>(thread_results.size()), 0l);
  for(int i = 0; i < thread_results.size(); ++i) {
    for(int tid = 0; tid < thread_results[i].size(); ++tid) {
      results[i] += thread_results[i][tid];
      net_winnings += thread_results[i][tid];
    }
  }
  assert(net_winnings == 0l && "Net winnings are not zero sum.");
  return results;
}

std::vector<long> simulate_round(Board board, std::vector<Hand> hands, ActionHistory actions, int n_chips, int ante) {
  omp::HandEvaluator eval;
  PokerState state(hands.size(), n_chips, ante);
  if(verbose) {
    for(int i = 0; i < hands.size(); ++i) {
      std::cout << "Player " << i << ": " << cards_to_str(hands[i].cards().data(), 2) << "\n";
    }
    std::cout << "Board: " << cards_to_str(board.cards().data(), 5) << "\n";
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
