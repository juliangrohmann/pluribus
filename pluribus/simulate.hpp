#pragma once

#include <vector>
#include <pluribus/poker.hpp>
#include <pluribus/agent.hpp>

namespace pluribus {

std::vector<std::vector<long>> simulate(std::vector<Agent*> agents, int n_chips, int ante, unsigned long n_iter);
std::vector<long> simulate_round(Board board, std::vector<Hand> hands, ActionHistory actions, int n_chips, int ante);

}
