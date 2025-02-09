#pragma once

#include <string>
#include <omp/Hand.h>

namespace pluribus {

int card_to_idx(const std::string& card);
std::string idx_to_card(int idx);

}