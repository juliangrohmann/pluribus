#include <pluribus/constants.hpp>
#include <pluribus/dist.hpp>
#include <pluribus/range.hpp>

#include "range_viewer.hpp"

namespace pluribus {

PokerRange build_distribution(const long n, const std::function<void(PokerRange&)>& sampler,
    const bool verbose) {
  PokerRange dist;
  const long log_interval = n / 10;
  for(long i = 0; i < n; ++i) {
    if(verbose && i % log_interval == 0) {
      std::cout << std::fixed << std::setprecision(0) << "Build distribution: " << i / static_cast<double>(n) * 100.0 << "%\n";
    }
    sampler(dist);
  }
  dist.normalize();
  return dist;
}

void distribution_to_png(const PokerRange& dist, const std::string& fn) {
  PokerRange range = dist;
  range.make_relative();
  std::cout << range.to_string() << "\n";
  auto renderer = PngRangeViewer(fn);
  renderer.render({RenderableRange{range, "Hand Distribution", Color::RED, false}});
}

double distribution_rmse(const PokerRange& dist_1, const PokerRange& dist_2) {
  auto norm_dist_1 = dist_1;
  auto norm_dist_2 = dist_2;
  norm_dist_1.normalize();
  norm_dist_2.normalize();
  double rmse = 0.0;
  for(int i = 0; i < MAX_COMBOS; ++i) {
    Hand hand = HoleCardIndexer::get_instance()->hand(i);
    rmse += pow(norm_dist_1.frequency(hand) - norm_dist_2.frequency(hand), 2);
  }
  return pow(rmse, 0.5);
}

}
