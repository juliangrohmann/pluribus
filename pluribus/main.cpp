#include <iostream>
#include <map>
#include <functional>
#include <cstdlib>
#include <pluribus/poker.hpp>
#include <pluribus/cluster.hpp>
#include <pluribus/range_viewer.hpp>
#include <pluribus/traverse.hpp>

using namespace pluribus;

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
    if(argc > 3 && strcmp(argv[2], "--png") == 0) {
      if(argc <= 4) std::cout << "Missing filename.\n";
      else {
        PngRangeViewer viewer{argv[3]};
        traverse(&viewer, argv[4]);
      }
    }
    else {
      WindowRangeViewer viewer{"traverse"};
      traverse(&viewer, argv[2]);
    }
    
  }
  else {
    std::cout << "Unknown command." << std::endl;
  }

  return 0;
}
