#include <iostream>
#include <pluribus/poker.hpp>
#include <pluribus/cluster.hpp>
#include <pluribus/range_viewer.hpp>
#include <pluribus/traverse.hpp>
#include <pluribus/blueprint.hpp>

using namespace pluribus;

void traverse_strategy(RangeViewer* viewer_p, const std::string& fn, const std::string& type) {
  if(type == "--blueprint"){
    Logger::log("Traversing blueprint: " + fn);
    traverse_blueprint(viewer_p, fn);
  }
  else if(type == "--tree") {
    Logger::log("Traversing tree: " + fn);
    traverse_tree(viewer_p, fn);
  }
  else {
    Logger::log("Unknown traverse type: " + type);
  }
}

int main(int argc, char* argv[]) {
  if(argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <command>" << std::endl;
    return 1;
  }

  if(std::string command = argv[1]; command == "cluster") {
    // ./Pluribus cluster round
    if(int round = atoi(argv[2]); round < 1 || round > 3) {
      std::cout << "1 <= round <= 3 required. Given: " << round << std::endl;
    }
    else {
      std::cout << "Clustering round " << round << "..." << std::endl;
      build_ochs_features(round);
    }
  }
  else if(command == "traverse") {
    // ./Pluribus traverse --blueprint --png out.png lossless_bp_fn
    // ./Pluribus traverse --trainer --png out.png lossless_bp_fn
    if(argc > 4 && strcmp(argv[3], "--png") == 0) {
      PngRangeViewer viewer{argv[4]};
      traverse_strategy(&viewer, argv[5], argv[2]);
    }
    else {
      WindowRangeViewer viewer{"traverse"};
      traverse_strategy(&viewer, argv[3], argv[2]);
    }
    
  }
  else if(command == "blueprint") {
    // ./Pluribus blueprint final_snapshot_fn snapshot_dir buf_dir out_fn
    if(argc < 6) {
      std::cout << "Missing arguments to build blueprint.\n";
    }
    else {
      LosslessBlueprint lossless_bp;
      lossless_bp.build(argv[2], get_filepaths(argv[3]), argv[4]);
      std::string lossless_fn = "lossless_" + std::string{argv[5]};
      cereal_save(lossless_bp, lossless_fn);
      SampledBlueprint sampled_bp;
      sampled_bp.build(lossless_fn, argv[2], argv[4]);
      cereal_save(sampled_bp, "sampled_" + std::string{argv[5]});

    }
  }
  else if(command == "blueprint-cached") {
    // ./Pluribus blueprint preflop_buf_fn final_snapshot_fn buf_dir out_fn
    if(argc < 6) {
      std::cout << "Missing arguments to build blueprint from cache.\n";
    }
    else {
      LosslessBlueprint lossless_bp;
      lossless_bp.build_cached(argv[2], argv[3], get_filepaths(argv[4]));
      std::string lossless_fn = "lossless_" + std::string{argv[5]};
      cereal_save(lossless_bp, lossless_fn);
      SampledBlueprint sampled_bp;
      sampled_bp.build(lossless_fn, argv[3], argv[4]);
      cereal_save(sampled_bp, "sampled_" + std::string{argv[5]});

    }
  }
  else if(command == "sampled-blueprint") {
    // ./Pluribus sampled-blueprint lossless_bp_fn final_snapshot_fn buf_dir out_fn
    if(argc < 6) {
      std::cout << "Missing arguments to build blueprint.\n";
    }
    else {
      SampledBlueprint sampled_bp;
      sampled_bp.build(argv[2], argv[3], argv[4]);
      cereal_save(sampled_bp, "sampled_" + std::string{argv[5]});

    }
  }
  else {
    std::cout << "Unknown command." << std::endl;
  }

  return 0;
}
