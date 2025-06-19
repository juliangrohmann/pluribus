#pragma once

#include <iostream>
#include <random>
#include <thread>
#include <memory>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <omp/Random.h>

namespace pluribus {

class GlobalRNG {
public:
  static std::mt19937& instance() {
    // thread_local std::mt19937 rng(42);
    thread_local std::mt19937 rng(std::random_device{}());
    return rng;
  }

  static double uniform() {
    thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(instance());
  }
};

class GSLGlobalRNG {
public:
  static gsl_rng*& instance() {
    thread_local gsl_rng* rng = nullptr;
    if(!rng) {
      rng = gsl_rng_alloc(gsl_rng_mt19937);
      gsl_rng_set(rng, static_cast<unsigned long>(std::time(nullptr) + std::hash<std::thread::id>{}(std::this_thread::get_id())));
    }
    return rng;
  }

  static double uniform() {
    return gsl_rng_uniform(instance());
  }

  static void cleanup() {
    if(instance()) {
      gsl_rng_free(instance());
      instance() = nullptr;
    }
  }

  ~GSLGlobalRNG() {
    cleanup();
  }
};

class GSLDiscreteDist {
public:
  GSLDiscreteDist(const std::vector<double>& weights) { 
    _dist = std::shared_ptr<gsl_ran_discrete_t>{gsl_ran_discrete_preproc(weights.size(), weights.data()), [](gsl_ran_discrete_t* p) { gsl_ran_discrete_free(p); }}; 
  }
  size_t sample() const { return gsl_ran_discrete(GSLGlobalRNG::instance(), _dist.get()); }
  ~GSLDiscreteDist() {  }
private:
  std::shared_ptr<gsl_ran_discrete_t> _dist = nullptr;
};

}
