#pragma once

#include <atomic>

namespace pluribus {

class SpinLock {
public:
  explicit SpinLock() : flag{ATOMIC_FLAG_INIT} {}
  void lock() {
    while (flag.test_and_set(std::memory_order_acquire)) { /* spin */ }
  }
  void unlock() {
    flag.clear(std::memory_order_release);
  }
private:
  std::atomic_flag flag;
};

}

template <>
class std::lock_guard<pluribus::SpinLock> {
public:
  explicit lock_guard(pluribus::SpinLock& m) : _m(m) { _m.lock(); }
  ~lock_guard() { _m.unlock(); }
private:
  pluribus::SpinLock& _m;
};
