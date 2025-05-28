#include <iostream>
#include <map>
#include <functional>
#include <cstdlib>
#include <pluribus/poker.hpp>
#include <pluribus/cluster.hpp>
#include <pluribus/range_viewer.hpp>
#include <pluribus/traverse.hpp>
#include <pluribus/blueprint.hpp>

using namespace pluribus;

void traverse_strategy(RangeViewer* viewer_p, std::string fn, bool trainer) {
  if(trainer) {
    traverse_trainer(viewer_p, fn);
  }
  else {
    traverse_blueprint(viewer_p, fn);
  }
}

int main(int argc, char* argv[]) {
  if(argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <command>" << std::endl;
    return 1;
  }

  std::string command = argv[1];

  if(command == "cluster") {
    int round = atoi(argv[2]);
    if(round < 1 || round > 3) {
      std::cout << "1 <= round <= 3 required. Given: " << round << std::endl;
    }
    else {
      std::cout << "Clustering round " << round << "..." << std::endl;
      build_ochs_features(round);
    }
  }
  else if(command == "traverse") {
    bool trainer = strcmp(argv[2], "--trainer") == 0;
    if(argc > 4 && strcmp(argv[3], "--png") == 0) {
      if(argc <= 4) std::cout << "Missing filename.\n";
      else {
        PngRangeViewer viewer{argv[4]};
        traverse_strategy(&viewer, argv[5], trainer);
      }
    }
    else {
      WindowRangeViewer viewer{"traverse"};
      traverse_strategy(&viewer, argv[3], trainer);
    }
    
  }
  else if(command == "blueprint") {
    if(argc < 6) {
      std::cout << "Missing arguments to build blueprint.\n";
    }
    else {
      LosslessBlueprint lossless_bp;
      lossless_bp.build(argv[2], get_filepaths(argv[3]), argv[4]);
      std::string lossless_fn = "lossless_" + std::string{argv[5]};
      cereal_save(lossless_bp, lossless_fn);
      SampledBlueprint sampled_bp{lossless_fn, argv[4]};
      cereal_save(sampled_bp, "sampled_" + std::string{argv[5]});

    }
  }
  else if(command == "sampled-blueprint") {
    if(argc < 5) {
      std::cout << "Missing arguments to build blueprint.\n";
    }
    else {
      SampledBlueprint sampled_bp{argv[2], argv[3]};
      cereal_save(sampled_bp, "sampled_" + std::string{argv[4]});

    }
  }
  else {
    std::cout << "Unknown command." << std::endl;
  }

  return 0;
}
