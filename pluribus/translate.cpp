#include <random>
#include <pluribus/rng.hpp>
#include <pluribus/translate.hpp>

namespace pluribus {

std::vector<std::pair<Action, double>> translatable_actions(const SlimPokerState& state, const std::vector<Action>& actions) {
  std::vector<std::pair<Action, double>> translatable;
  for(Action a : actions) {
    if(a == Action::ALL_IN) translatable.emplace_back(a, fractional_bet_size(state, total_bet_size(state, Action::ALL_IN)));
    else if(a.get_bet_type() >= 0.0) translatable.emplace_back(a, a.get_bet_type());
  }
  std::ranges::sort(translatable, std::ranges::less{}, [](const auto& e) { return e.second; });
  return translatable;
}

TranslationResult pseudo_harmonic_result(const Action a, const std::vector<Action>& actions, const SlimPokerState& state) {
  if(a == Action::FOLD) return {Action::FOLD, Action::FOLD, 1.0};
  const double x = a != Action::ALL_IN ? a.get_bet_type() : fractional_bet_size(state, total_bet_size(state, Action::ALL_IN));
  if(x < 0.0) return {a, a, 1.0};
  const auto translatable = translatable_actions(state, actions);
  for(int i = 0; i < translatable.size(); ++i) {
    const Action action = translatable[i].first;
    const double bet_size = translatable[i].second;
    if(bet_size == x) return {action, action, 1.0};
    if(bet_size > x) {
      if(i == 0) return {action, action, 1.0};;
      const double A = translatable[i - 1].second;
      const double B = translatable[i].second;
      const double p_A = (B - x) * (1 + A) / ((B - A) * (1 + x));
      return {translatable[i - 1].first, translatable[i].first, p_A};
    }
  }
  const Action max_action = translatable[translatable.size() - 1].first;
  return {max_action, max_action, 1.0};
}

Action sample(const TranslationResult& result) {
  return GSLGlobalRNG::uniform() < result.p_A ? result.A : result.B;
}

Action translate_pseudo_harmonic(const Action a, const std::vector<Action>& actions, const SlimPokerState& state) {
  return sample(pseudo_harmonic_result(a, actions, state));
}

}
