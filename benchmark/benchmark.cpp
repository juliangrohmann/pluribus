#include <iostream>
#include <string>
#include <set>
#include <array>
#include <fstream>

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <omp/Hand.h>
#include <omp/HandEvaluator.h>
#include <pluribus/poker.hpp>

using namespace pluribus;
using std::string;

TEST_CASE("Evaluate benchmark", "[eval]") {
  omp::HandEvaluator evaluator;
  omp::Hand hero = omp::Hand("Qd") + omp::Hand("As") + omp::Hand("6h") + omp::Hand("Js") + omp::Hand("2c");
  omp::Hand villain = omp::Hand("3d") + omp::Hand("9h") + omp::Hand("Kc") + omp::Hand("4h") + omp::Hand("8s");
  BENCHMARK("Eval 5 cards") {
    bool winner = evaluator.evaluate(hero) > evaluator.evaluate(villain);
  };
  
  hero += omp::Hand("Jd");
  villain += omp::Hand("4c");
  BENCHMARK("Eval 6 cards") {
    bool winner = evaluator.evaluate(hero) > evaluator.evaluate(villain);
  };

  hero += omp::Hand("6c");
  villain += omp::Hand("Qc");
  BENCHMARK("Eval 7 cards") {
    bool winner = evaluator.evaluate(hero) > evaluator.evaluate(villain);
  };
};