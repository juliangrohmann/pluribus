#include <pluribus/blueprint.hpp>

using namespace pluribus;

int main(const int argc, char* argv[]) {
  if(argc < 5) {
    std::cout << "Usage: ./Convert lossless_fn sampled_fn n_snapshots n_iterations\n";
    return 1;
  }
  const int n_snapshots = atoi(argv[3]);
  const long n_iterations = atol(argv[4]);

  std::cout << "n_snapshots=" << n_snapshots << "\n";
  std::cout << "n_iterations=" << n_iterations << "\n";

  LosslessBlueprint lossless;
  cereal_load(lossless, argv[1]);
  lossless.set_n_snapshots(n_snapshots);
  lossless.set_n_iterations(n_iterations);
  std::cout << "converted n_snapshots=" << lossless.get_n_snapshots() << "\n";
  std::cout << "converted n_iterations=" << lossless.get_n_iterations() << "\n";
  cereal_save(lossless, std::string{argv[1]} + ".converted");

  SampledBlueprint sampled;
  cereal_load(sampled, argv[2]);
  cereal_save(sampled, std::string{argv[2]} + ".converted");
}
