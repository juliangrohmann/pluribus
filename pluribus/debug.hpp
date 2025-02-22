#pragma once

#include <string>
#include <cstdlib>
#include <unordered_map>
#include <hand_isomorphism/hand_index.h>
#include <pluribus/mccfr.hpp>
#include <pluribus/actions.hpp>
#include <pluribus/poker.hpp>

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
void print_cluster(int cluster, int round, int n_clusters);
void print_similar_boards(std::string board, int n_clusters=200);
std::string strategy_str(const std::unordered_map<InformationSet, std::unordered_map<Action, StrategyState>>& strategy, 
                         const ActionHistory& history, Action action, const Board& board, int round);
void evaluate_strategy(const StrategyMap& hero, const StrategyMap& villain);

}