#pragma once

#include <string>
#include <cstdlib>
#include <unordered_map>
#include <hand_isomorphism/hand_index.h>
#include <pluribus/mccfr.hpp>
#include <pluribus/actions.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/agent.hpp>

namespace pluribus {

constexpr bool is_debug = 
#ifdef NDEBUG
  false;
#else
  true;
#endif

constexpr bool verbose = is_debug &
#ifdef VERBOSE
  true;
#else
  false;
#endif

std::string round_to_str(int round);
std::string pos_to_str(int pos, int n_players);
void print_cluster(int cluster, int round, int n_clusters);
void print_similar_boards(std::string board, int n_clusters=200);
std::string strategy_str(const BlueprintTrainer& trainer, const PokerState& state, Action action, const Board& board);
void evaluate_agents(const std::vector<Agent*>& agents, const PokerConfig& config, long n_iter);
void evaluate_strategies(const std::vector<BlueprintTrainer*>& strategies, long n_iter);
void evaluate_vs_random(const BlueprintTrainer* hero, long n_iter);

}