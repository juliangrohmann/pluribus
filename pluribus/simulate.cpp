#include <cassert>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <stack>
#include <omp/HandEvaluator.h>
#include <pluribus/debug.hpp>
#include <pluribus/mccfr.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/simulate.hpp>
#include <pluribus/util.hpp>

namespace pluribus {

std::vector<long> get_net_payoffs(const PokerState& state, const Board& board, const std::vector<Hand>& hands, int n_chips, const omp::HandEvaluator& eval) {
  const RakeStructure no_rake{0.0, 0};
  std::vector<long> net_payoffs;
  for(int i = 0; i < state.get_players().size(); ++i) {
    net_payoffs.push_back(utility(state, i, board, hands, n_chips, no_rake, eval));
  }
  return net_payoffs;
}

std::vector<long> simulate(const std::vector<Agent*>& agents, const PokerConfig& config, const int n_chips, const long n_iter) {
  std::vector<std::vector<long>> thread_results{agents.size()};
  const int ntid = omp_get_max_threads();
  const long log_interval = n_iter / ntid / 100;
  for(auto& result : thread_results) result.resize(ntid);

  #pragma omp parallel for schedule(static)
  for(long t = 0; t < n_iter; ++t) {
    thread_local omp::HandEvaluator eval;
    thread_local Deck deck;
    thread_local Board board;
    thread_local std::vector<Hand> hands{agents.size()};
    const int tid = omp_get_thread_num();
    if(tid == 0 && t % log_interval == 0) std::cout << "Sim: " << std::setprecision(1) << std::fixed << t / static_cast<double>(log_interval) << "%\n";
    PokerState state(config, n_chips);
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
      state = state.apply(agents[state.get_active()]->act(state, board, hands[state.get_active()]));
    }

    std::vector<long> payoffs = get_net_payoffs(state, board, hands, n_chips, eval);
    long net_round = 0l;
    for(int i = 0; i < hands.size(); ++i) {
      thread_results[i][tid] += payoffs[i];
      net_round += payoffs[i];
    }
    assert(net_round == 0l && "Round winnings are not zero sum.");
  }

  long net_winnings = 0l;
  std::vector results(static_cast<int>(thread_results.size()), 0l);
  for(int i = 0; i < thread_results.size(); ++i) {
    for(int tid = 0; tid < thread_results[i].size(); ++tid) {
      results[i] += thread_results[i][tid];
      net_winnings += thread_results[i][tid];
    }
  }
  assert(net_winnings == 0l && "Net winnings are not zero sum.");
  return results;
}

std::vector<long> simulate_round(const Board& board, const std::vector<Hand>& hands, const ActionHistory& actions, const PokerConfig& config,
    const int n_chips) {
  const omp::HandEvaluator eval;
  PokerState state(config, n_chips);
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
  std::vector<long> results = get_net_payoffs(state, board, hands, n_chips, eval);
  int net_winnings = 0;
  for(int i = 0; i < hands.size(); ++i) {
    results[i] += state.get_players()[i].get_chips() - n_chips;
    if(verbose) {
      std::cout << std::setprecision(2) << std::fixed << "Player: " << i << ": "
          << std::setw(8) << std::showpos << (results[i] / 100.0) << " bb\n" << std::noshowpos;
    }
    net_winnings += results[i];
  }
  assert(net_winnings == 0 && "Winnings are not zero sum.");
  return results;
}

}
