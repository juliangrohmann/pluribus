#include <chrono>
#include <cnpy.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <omp.h>
#include <string>
#include <vector>
#include <cereal/types/array.hpp>
#include <hand_isomorphism/hand_index.h>
#include <omp/CardRange.h>
#include <omp/EquityCalculator.h>
#include <pluribus/cluster.hpp>
#include <pluribus/constants.hpp>
#include <pluribus/util.hpp>
#include <tqdm/tqdm.hpp>

namespace pluribus {

void assign_features(const std::string& hand, const std::string& board, float* data) {
  const omp::Hand board_hand = omp::Hand::empty() + omp::Hand(board);
  for(int i = 0; i < 8; ++i) {
    *(data + i) = equity(omp::Hand(hand), omp::CardRange(ochs_categories[i]), board_hand);
  }
}

std::array<double, 2> eval(const omp::Hand& hero, const omp::CardRange& vill_rng, const omp::Hand& board) {
  const omp::HandEvaluator evaluator;
  std::array<double, 2> results = {0, 0};

  const int hero_val = evaluator.evaluate(hero + board);
  const omp::Hand dead_cards = hero + board;
  for(const auto& combo : vill_rng.combinations()) {
    omp::Hand villain = omp::Hand(combo[0]) + omp::Hand(combo[1]);
    if(dead_cards.contains(villain)) {
      continue;
    }
    if(const int vill_val = evaluator.evaluate(villain + board); hero_val > vill_val) results[0] += 1;
    else if(hero_val < vill_val) results[1] += 1;
    else {
      results[0] += 0.5;
      results[1] += 0.5;
    }
  }
  return results;
}

std::array<double, 2> enumerate(const omp::Hand& hero, const omp::CardRange& villain, const omp::Hand& board) {
  if(board.count() == 5) {
    return eval(hero, villain, board);
  }
  std::array<double, 2> results = {0, 0};
  for(int idx = 0; idx < MAX_CARDS; ++idx) {
    auto card = omp::Hand(idx);
    if(hero.contains(card) || board.contains(card)) continue;
    auto tmp_res = enumerate(hero, villain, board + card);
    results[0] += tmp_res[0];
    results[1] += tmp_res[1];
  }
  return results;
}

double equity(const omp::Hand& hero, const omp::CardRange& villain, const omp::Hand& board) {
  const auto results = enumerate(hero, villain, board);
  return results[0] / (results[0] + results[1]);
}

void solve_features(const hand_indexer_t& indexer, const int round, const int card_sum, const size_t start, const size_t end, const std::string& fn) {
  const int num_threads = omp_get_max_threads();
  const size_t chunk_size = std::max((end - start) / num_threads, 1ul);
  const size_t log_interval = std::max(chunk_size / 1000, 1ul);
  std::cout << start << " <= idx < " << end << std::endl;
  std::cout << "num_threads = " << num_threads << std::endl;
  std::cout << "chunk_size = " << chunk_size << std::endl;
  std::cout << "log_interval = " << log_interval << std::endl;
  std::cout << "allocating feature map..." << std::endl;
  std::vector<float> feature_map((end - start) * 8);
  std::cout << "launching threads..." << std::endl;
  const std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  
  #pragma omp parallel for schedule(static)
  for(size_t idx = start; idx < end; ++idx) {
    if(const int tid = omp_get_thread_num(); tid == 0 && idx % log_interval == 0) {
      std::chrono::steady_clock::time_point t_f = std::chrono::steady_clock::now();
      const auto dt = std::chrono::duration_cast<std::chrono::seconds>(t_f - begin).count();
      const double p = static_cast<double>(idx) / chunk_size;
      std::cout << std::right << " (round " << round << ") " << std::setw(11) << std::to_string(idx) << ":   " 
                << std::fixed << std::setprecision(1) << std::setw(5) << (p * 100) << "%"
                << "    Elapsed: " << std::setw(7) << std::setprecision(0) << dt << " s"
                << "    Remaining: " << std::setw(7) << 1 / p * dt - dt << " s" << std::endl;
    }

    uint8_t cards[7] = {};
    hand_unindex(&indexer, round, idx, cards);
    std::string hand = "";
    std::string board = "";
    for(int i = 0; i < 2; ++i) {
      hand += idx_to_card(cards[i]);
    }
    for(int i = 2; i < card_sum; ++i) {
      board += idx_to_card(cards[i]);
    }
    assign_features(hand, board, &feature_map[(idx - start) * 8]);
  }

  std::cout << "len = " << feature_map.size() << std::endl;
  std::cout << "writing features..." << std::endl;
  cnpy::npy_save(fn, feature_map.data(), {(end - start), 8}, "w");
}

void build_ochs_features(const int round) {
  omp::EquityCalculator eq;
  hand_indexer_t indexer;
  const int card_sum = init_indexer(indexer, round);
  const size_t n_idx = hand_indexer_size(&indexer, round);
  if(round == 3) {
    constexpr int n_batches = 10;
    const size_t batch_size = n_idx / n_batches;
    std::cout << "n_batches = " + n_batches << std::endl;
    std::cout << "batch_size = " + batch_size << std::endl;
    for(int batch = 0; batch < n_batches; ++batch) {
      std::cout << "launching batch " << batch << std::endl;
      solve_features(indexer, round, card_sum, batch * batch_size, batch == n_batches - 1 ? n_idx : (batch + 1) * batch_size,
          std::string("features_") + std::to_string(round) + "_b" + std::to_string(batch) + ".npy");
    }
  }
  else {
    solve_features(indexer, round, card_sum, 0, n_idx, std::string("features_") + std::to_string(round) + ".npy");
  }
  hand_indexer_free(&indexer);
}

std::string cluster_filename(const int round, const int n_clusters, const int split) {
  const std::string base = "clusters_r" + std::to_string(round) + "_c" + std::to_string(n_clusters);
  return base + (round == 3 ? "_p" + std::to_string(split) + ".npy": ".npy");
}

std::vector<uint16_t> load_clusters(const int round, const int n_clusters, const int split) {
  return cnpy::npy_load(cluster_filename(round, n_clusters, split)).as_vec<uint16_t>();
}

std::array<std::vector<uint16_t>, 4> init_flat_cluster_map(const int n_clusters) {
  std::cout << "Initializing flat cluster map (n_clusters=" << n_clusters << ")...\n";
  std::array<std::vector<uint16_t>, 4> cluster_map;
  cluster_map[0].resize(169);
  std::iota(cluster_map[0].begin(), cluster_map[0].end(), 0);
  for(int i = 1; i <= 3; ++i) {
    std::cout << "(Flat: " << n_clusters << " clusters) Loading round " << i << "...\n";
    cluster_map[i] = load_clusters(i, n_clusters, 1);
  }
  auto s2 = cnpy::npy_load(cluster_filename(3, n_clusters, 2)).as_vec<uint16_t>();
  const size_t s1_size = cluster_map[3].size();
  cluster_map[3].resize(cluster_map[3].size() + s2.size());
  std::ranges::copy(s2, cluster_map[3].data() + s1_size);
  std::cout << "Loaded all clusters.\n";
  return cluster_map;
}

std::unique_ptr<FlatClusterMap> FlatClusterMap::_instance = nullptr;

FlatClusterMap::FlatClusterMap() {
  _cluster_map = init_flat_cluster_map(200);
}

}