#pragma once

#include <string>
#include <cstdlib>
#include <hand_isomorphism/hand_index.h>

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

}
