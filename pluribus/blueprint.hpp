#pragma once

#include <string>
#include <vector>
#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>
#include <pluribus/debug.hpp>
#include <pluribus/rng.hpp>
#include <pluribus/cereal_ext.hpp>
#include <pluribus/storage.hpp>
#include <pluribus/mccfr.hpp>

namespace pluribus {

template <class T>
class Blueprint : public Strategy<T> {
public:
  Blueprint() : _freq{nullptr} {}

  const StrategyStorage<T>& get_strategy() const {
    if(_freq) return *_freq;
    throw std::runtime_error("Blueprint strategy is null.");
  }
  const BlueprintTrainerConfig& get_config() const { return _config; }
  void set_config(BlueprintTrainerConfig config) { _config = config; }

  template <class Archive>
  void serialize(Archive& ar) {
    ar(_freq);
  }

protected:
  void assign_freq(StrategyStorage<T>* freq) { _freq = std::unique_ptr<StrategyStorage<T>>{freq}; }
  std::unique_ptr<StrategyStorage<T>>& get_freq() { return _freq; }

private:
  std::unique_ptr<StrategyStorage<T>> _freq;
  BlueprintTrainerConfig _config;
};

class LosslessBlueprint : public Blueprint<float> {
public:
  void build(const std::string& preflop_fn, const std::vector<std::string>& postflop_fns, const std::string& buf_dir = "");
  double enumerate_ev(const PokerState& state, int i, const std::vector<PokerRange>& ranges, const std::vector<uint8_t>& board) const;
  double monte_carlo_ev(int n, const PokerState& state, int i, const std::vector<PokerRange>& ranges, const std::vector<uint8_t>& board) const;

private:
  double node_ev(const PokerState& state, int i, const std::vector<Hand>& hands, const std::vector<uint8_t>& board, int stack_size, std::vector<CachedIndexer>& indexers, const omp::HandEvaluator& eval) const;
};

std::vector<float> biased_freq(const std::vector<Action>& actions, const std::vector<float>& freq, Action bias, float factor);
void _validate_ev_inputs(const PokerState& state, int i, const std::vector<PokerRange>& ranges, const std::vector<uint8_t>& board);

class SampledBlueprint : public Blueprint<Action> {
public:
  void build(const std::string& lossless_bp_fn, const std::string& buf_dir, float bias_factor = 5.0f);
  double monte_carlo_ev(int n, const std::vector<Action>& biases, const PokerState& state, int i, const std::vector<PokerRange>& ranges, const std::vector<uint8_t>& board) const;
};

template <class T>
class _ActionProvider {
public:
  virtual Action next_action(const PokerState& state, const std::vector<Hand>& hands, const Board& board, const Blueprint<T>& bp) const = 0;
};

template <class T>
double _monte_carlo_ev(int n, const PokerState& init_state, int i, const std::vector<PokerRange>& ranges, const std::vector<uint8_t>& init_board, int stack_size, const _ActionProvider<T>& action_provider, const Blueprint<T>& bp) {
  _validate_ev_inputs(init_state, i, ranges, init_board);

  std::vector<double> weights;
  std::vector<std::vector<Hand>> outcomes;
  for(const auto& oop : ranges[0].hands()) {
    for(const auto& ip : ranges[1].hands()) {
      if(collides(oop, ip) || collides(oop, init_board) || collides(ip, init_board)) continue;
      weights.push_back(ranges[0].frequency(oop) * ranges[1].frequency(ip));
      outcomes.push_back({oop, ip});
    }
  }
  std::discrete_distribution<> dist{weights.begin(), weights.end()};

  Deck deck;
  omp::HandEvaluator eval;
  long value = 0;
  for(int t = 0; t < n; ++t) {
    deck.clear_dead_cards();
    std::vector<Hand> hands = outcomes[dist(GlobalRNG::instance())];
    for(const auto& hand : hands) deck.add_dead_cards(hand.cards());
    deck.add_dead_cards(init_board);
    deck.shuffle();
    
    auto board_cards = init_board;
    while(board_cards.size() < 5) board_cards.push_back(deck.draw());
    Board board{board_cards};

    PokerState state = init_state;
    while(!state.is_terminal()) {
      state = state.apply(action_provider.next_action(state, hands, board, bp));
    }
    int u = utility(state, i, board, hands, stack_size, eval);
    value += u;
  }
  return value / static_cast<double>(n);
}

}
