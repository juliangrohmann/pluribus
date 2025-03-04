#pragma once

#include <vector>
#include <pluribus/poker.hpp>
#include <pluribus/agent.hpp>

namespace pluribus {

std::vector<long> simulate(const std::vector<Agent*>& agents, const PokerConfig& config, long n_iter);
std::vector<long> simulate_round(const Board& board, const std::vector<Hand>& hands, const ActionHistory& actions, const PokerConfig& config);

}
