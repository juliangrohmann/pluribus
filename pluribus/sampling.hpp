#pragma once 

#include <omp/Random.h>
#include <pluribus/constants.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/rng.hpp>
#include <pluribus/range.hpp>

namespace pluribus {

struct RoundSample {
  std::vector<Hand> hands;
  double weight = 1.0;
  uint64_t mask = 0L;
};

class SamplingAlgorithm {
public:
  SamplingAlgorithm(uint64_t init_mask = 0L);
  SamplingAlgorithm(const std::vector<uint8_t>& dead_cards);

  virtual RoundSample sample() = 0;

protected:
  uint64_t init_mask() const { return _init_mask; }

private:
  uint64_t _init_mask = 0L;
};

class MarginalRejectionSampler : public SamplingAlgorithm {
public:
  MarginalRejectionSampler(const std::vector<PokerRange>& ranges, const std::vector<uint8_t>& dead_cards = {});

  RoundSample sample() override;

private:
  std::vector<GSLDiscreteDist> _hand_dists;
  int _hand_idxs[MAX_PLAYERS];
};

class ImportanceSampler : public SamplingAlgorithm {
public:
  ImportanceSampler(const std::vector<PokerRange>& ranges, const std::vector<uint8_t>& dead_cards = {});

protected:
  const std::vector<std::vector<Hand>>& filtered_hands() const { return _filt_hands; };
  const std::vector<std::vector<double>>& filtered_weights() const { return _filt_weights; };
  omp::XoroShiro128Plus& rng() { return _rng; }
  double joint_probability() const { return _joint_prob; }
  int* hand_indexes() { return _hand_idxs; };

private:
  std::vector<std::vector<Hand>> _filt_hands;
  std::vector<std::vector<double>> _filt_weights;
  omp::XoroShiro128Plus _rng;
  double _joint_prob;
  int _hand_idxs[MAX_PLAYERS];
};

class ImportanceRejectionSampler : public ImportanceSampler {
public:
  ImportanceRejectionSampler(const std::vector<PokerRange>& ranges, const std::vector<uint8_t>& dead_cards = {});

  RoundSample sample() override;

protected:
  RoundSample sample_hands();

private:
  std::vector<omp::FastUniformIntDistribution<unsigned, 21>> _uniform_dists;
};

class ImportanceRandomWalkSampler : public ImportanceRejectionSampler {
public:
  ImportanceRandomWalkSampler(const std::vector<PokerRange>& ranges, const std::vector<uint8_t>& dead_cards = {});

  RoundSample sample() override;
  void next_sample(RoundSample& sample);

private:
  omp::FastUniformIntDistribution<unsigned, 16> _idx_dist;
};

enum class SamplingMode {
  AUTOMATIC, MARGINAL_REJECTION, IMPORTANCE_REJECTION, IMPORTANCE_RANDOM_WALK
};

class RoundSampler {
public:
  RoundSampler(const std::vector<PokerRange>& ranges, const std::vector<uint8_t>& dead_cards = {});

  RoundSample sample();
  void next_sample(RoundSample& sample);
  void set_mode(SamplingMode mode) { _mode = mode; }

private:
  MarginalRejectionSampler _marginal_rejection;
  ImportanceRejectionSampler _importance_rejection;
  ImportanceRandomWalkSampler _importance_walk;
  SamplingMode _mode = SamplingMode::AUTOMATIC;
};

Board sample_board(const std::vector<uint8_t>& init_board, uint64_t mask);

}
