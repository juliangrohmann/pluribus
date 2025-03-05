#pragma once

#include <string>
#include <array>
#include <vector>
#include <memory>
#include <omp/EquityCalculator.h>
#include <omp/Hand.h>
#include <hand_isomorphism/hand_index.h>
#include <pluribus/poker.hpp>

namespace pluribus {

const std::array<std::string, 8> ochs_categories = {
  "32,42,52,62,72,43,53,63,73,54,64,82o,83o,74o,65o",
  "82s,92,T2,J2o,83s,93,T3o,J3o,74s,84,94,T4o,75o,85o,95o,T5o",
  "T3s,T4s,65s,75s,85s,95s,T5s,76,86,96,T6,87,97,T7o,98,T8o",
  "22,J2s,Q2,K2,J3s,Q3,K3o,J4,Q4,K4o,J5,Q5,J6,Q6o,J7o,Q7o",
  "Q6s,T7s,J7s,Q7s,T8s,J8,Q8,T9,J9,Q9,JT,QTo,QJo",
  "33,44,55,A2,K3s,A3,K4s,A4,K5,A5,K6,A6,K7,A7o,K8,A8o,K9o",
  "66,77,A7s,A8s,K9s,A9,QTs,KT,AT,QJs,KJ,AJ,KQ,AQ,AK",
  "88,99,TT,JJ,QQ,KK,AA"
};


void assign_features(const std::string& hand, const std::string& board, float* data);
double equity(const omp::Hand& hero, const omp::CardRange villain, const omp::Hand& board);
void build_ochs_features(int round);
std::string cluster_filename(int round, int n_clusters, int split);
std::array<std::vector<uint16_t>, 4> init_flat_cluster_map(int n_clusters);

class FlatClusterMap {
public:
  uint16_t cluster(int round, uint64_t index) const { return _cluster_map[round][index]; }
  uint16_t cluster(int round, const Board& board, const Hand& hand) const;

  static FlatClusterMap* get_instance() {
    if(!_instance) {
      _instance = std::unique_ptr<FlatClusterMap>(new FlatClusterMap());
    }
    return _instance.get();
  }

  FlatClusterMap(const FlatClusterMap&) = delete;
  FlatClusterMap& operator=(const FlatClusterMap&) = delete;

private:
  FlatClusterMap();

  std::array<std::vector<uint16_t>, 4> _cluster_map;

  static std::unique_ptr<FlatClusterMap> _instance;
};

}
