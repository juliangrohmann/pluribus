#pragma once

#include <pluribus/poker.hpp>

namespace pluribus {

struct TranslationResult {
  Action A;
  Action B;
  double p_A;
};

TranslationResult pseudo_harmonic_result(Action a, const std::vector<Action>& actions, const PokerState& state);
Action sample(const TranslationResult& result);
Action translate_pseudo_harmonic(Action a, const std::vector<Action>& actions, const PokerState& state);

}
