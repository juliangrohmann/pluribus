#pragma once

#include <vector>
#include <pluribus/poker.hpp>
#include <pluribus/agent.hpp>

namespace pluribus {

std::vector<std::vector<long>> simulate(std::vector<Agent*> agents, int n_chips, int ante, unsigned long n_iter);
std::vector<long> simulate_round(std::array<uint8_t, 5> board, std::vector<std::array<uint8_t, 2>> hands, ActionHistory actions, int n_chips, int ante);

}
