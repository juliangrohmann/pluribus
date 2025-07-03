#pragma once

#include <string>

namespace pluribus {

constexpr bool is_debug = 
#ifdef NDEBUG
  false;
#else
  true;
#endif

// ReSharper disable once CppCompileTimeConstantCanBeReplacedWithBooleanConstant
constexpr bool verbose = is_debug &
#ifdef VERBOSE
  true;
#else
                           false;
#endif

std::string round_to_str(int round);
std::string pos_to_str(int idx, int n_players);
void print_cluster(int cluster, int round, int n_clusters);
void print_similar_boards(const std::string &board, int n_clusters=200);

}