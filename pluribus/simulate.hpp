#pragma once

#include <vector>
#include <pluribus/agent.hpp>
#include <pluribus/poker.hpp>

namespace pluribus {

std::vector<long> simulate(const std::vector<Agent*>& agents, const PokerConfig& config, int n_chips, long n_iter);
std::vector<long> simulate_round(const Board& board, const std::vector<Hand>& hands, const ActionHistory& actions, const PokerConfig& config, int n_chips);

}
