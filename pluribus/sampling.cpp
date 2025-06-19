#include "ransampl.h"
#include <pluribus/logging.hpp>
#include <pluribus/rng.hpp>
#include <pluribus/sampling.hpp>

namespace pluribus {

SamplingAlgorithm::SamplingAlgorithm(uint64_t init_mask) : _init_mask{init_mask} {}

SamplingAlgorithm::SamplingAlgorithm(const std::vector<uint8_t>& dead_cards) : _init_mask{card_mask(dead_cards)} {}

MarginalRejectionSampler::MarginalRejectionSampler(const std::vector<PokerRange>& ranges, const std::vector<uint8_t>& dead_cards) 
    : SamplingAlgorithm{dead_cards} {
  _hand_dists.reserve(ranges.size());
  for(const auto& r : ranges) {
    _hand_dists.emplace_back(r.weights());
  }
}

RoundSample sample_rejection(int n_players, uint64_t mask, int* indexes, std::function<int(int)> idx_sampler, std::function<Hand(int,int)> idx_to_hand) {
  RoundSample sample{std::vector<Hand>(n_players)};
  int coll;
  int tries = 0;
  do {
    ++tries;
    if(tries >= 10'000) Logger::error("Too many sample rejections. Rejections=" + std::to_string(tries));
    sample.mask = mask;
    coll = 0;
    for(int i = 0; i < n_players; ++i) {
      indexes[i] = idx_sampler(i);
      sample.hands[i] = idx_to_hand(i, indexes[i]);
      coll += (sample.mask & sample.hands[i].mask()) != 0;
      sample.mask |= sample.hands[i].mask();
    }
  } while(coll > 0);
  return sample;
}

RoundSample MarginalRejectionSampler::sample() {
  return sample_rejection(_hand_dists.size(), init_mask(), _hand_idxs, [this](int i) { return this->_hand_dists[i].sample(); }, 
      [](int i, int h_idx) { return HoleCardIndexer::get_instance()->hand(h_idx); });
}

ImportanceSampler::ImportanceSampler(const std::vector<PokerRange>& ranges, const std::vector<uint8_t>& dead_cards) 
    : SamplingAlgorithm{dead_cards}, _rng{std::random_device{}()}, _joint_prob{1.0} {
  _filt_hands.resize(ranges.size());
  _filt_weights.resize(ranges.size());
  for(int r_idx = 0; r_idx < ranges.size(); ++r_idx) {
    const auto& weights = ranges[r_idx].weights();
    double total_weight = 0.0;
    for(int h_idx = 0; h_idx < weights.size(); ++h_idx) {
      if(weights[h_idx] > 0.0) {
        _filt_hands[r_idx].push_back(HoleCardIndexer::get_instance()->hand(h_idx));
        _filt_weights[r_idx].push_back(weights[h_idx]);
        total_weight += weights[h_idx];
      }
    }
    for(auto& w : _filt_weights[r_idx]) {
      w /= total_weight;
    }
  }
  for(int i = 0; i < ranges.size(); ++i) {
    _joint_prob *= 1.0 / _filt_hands[i].size();
  }
}

ImportanceRejectionSampler::ImportanceRejectionSampler(const std::vector<PokerRange>& ranges, const std::vector<uint8_t>& dead_cards) 
    : ImportanceSampler{ranges, dead_cards} {
  _uniform_dists.reserve(ranges.size());
  for(int i = 0; i < ranges.size(); ++i) {
    _uniform_dists.push_back(omp::FastUniformIntDistribution<unsigned, 21>(0, filtered_hands()[i].size() - 1));
  }
}

RoundSample ImportanceRejectionSampler::sample_hands() {
  return sample_rejection(_uniform_dists.size(), init_mask(), hand_indexes(),
      [this](int i) { return this->_uniform_dists[i](this->rng()); }, [this](int i, int h_idx) { return filtered_hands()[i][h_idx]; });
}

RoundSample ImportanceRejectionSampler::sample() {
  auto sample = sample_hands();
  sample.weight = 1.0 / joint_probability();
  for(int i = 0; i < _uniform_dists.size(); ++i) {
    sample.weight *= filtered_weights()[i][hand_indexes()[i]];
  }
  return sample;
}

ImportanceRandomWalkSampler::ImportanceRandomWalkSampler(const std::vector<PokerRange>& ranges, const std::vector<uint8_t>& dead_cards)
    : ImportanceRejectionSampler{ranges, dead_cards} {
  _idx_dist = omp::FastUniformIntDistribution<unsigned, 16>(0, ranges.size() - 1);
  sample_hands();
}

RoundSample ImportanceRandomWalkSampler::sample() {
  RoundSample sample;
  sample.hands.resize(filtered_hands().size());
  sample.weight = 1.0 / joint_probability();
  sample.mask = init_mask();
  int p_idx = _idx_dist(rng());
  for(int i = 0; i < filtered_hands().size(); ++i) {
    if(i != p_idx) {
      auto hand = filtered_hands()[i][hand_indexes()[i]];
      sample.hands[i] = hand;
      sample.weight *= filtered_weights()[i][hand_indexes()[i]];
      sample.mask |= hand.mask();
    }
  }
  int& combo_idx = hand_indexes()[p_idx];
  do {
    if(combo_idx == 0) combo_idx = filtered_hands()[p_idx].size();
    --combo_idx;
  } while(sample.mask & filtered_hands()[p_idx][combo_idx].mask());
  sample.hands[p_idx] = filtered_hands()[p_idx][combo_idx];
  sample.weight *= filtered_weights()[p_idx][combo_idx];
  sample.mask |= filtered_hands()[p_idx][combo_idx].mask();
  return sample;
}

void ImportanceRandomWalkSampler::next_sample(RoundSample& sample) {
  int p_idx = _idx_dist(rng());
  int& combo_idx = hand_indexes()[p_idx];
  sample.weight /= filtered_weights()[p_idx][combo_idx];
  sample.mask -= filtered_hands()[p_idx][combo_idx].mask();
  do {
    if(combo_idx == 0) combo_idx = filtered_hands()[p_idx].size();
    --combo_idx;
  } while(sample.mask & filtered_hands()[p_idx][combo_idx].mask());
  sample.hands[p_idx] = filtered_hands()[p_idx][combo_idx];
  sample.weight *= filtered_weights()[p_idx][combo_idx];
  sample.mask |= filtered_hands()[p_idx][combo_idx].mask();
}

RoundSampler::RoundSampler(const std::vector<PokerRange>& ranges, const std::vector<uint8_t>& dead_cards) 
    : _marginal_rejection{ranges, dead_cards}, _importance_rejection{ranges, dead_cards}, _importance_walk{ranges, dead_cards} {}

RoundSample RoundSampler::sample() {
  switch(_mode) {
    case SamplingMode::AUTOMATIC: 
    case SamplingMode::IMPORTANCE_RANDOM_WALK: return _importance_walk.sample();
    case SamplingMode::IMPORTANCE_REJECTION: return _importance_rejection.sample();
    case SamplingMode::MARGINAL_REJECTION: return _marginal_rejection.sample();
    default: Logger::error("Unknown sampling mode.");
  }
}

void RoundSampler::next_sample(RoundSample& sample) {
  return _importance_walk.next_sample(sample);
}

}