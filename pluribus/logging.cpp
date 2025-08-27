#include <pluribus/logging.hpp>

namespace pluribus {

std::unique_ptr<Logger> Logger::_instance = nullptr;

std::string progress_str(const uint64_t idx, const uint64_t total, const std::chrono::high_resolution_clock::time_point& t_0) {
  const auto dt = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - t_0).count();
  const double percent = static_cast<double>(idx) / static_cast<double>(total);
  std::ostringstream oss;
  oss << std::setw(11) << std::to_string(idx) << ":   "
      << std::fixed << std::setprecision(1) << std::setw(5) << (percent * 100) << "%"
      << std::setw(7) << std::setprecision(0) << idx / dt << "it/s    "
      << std::setw(7) << dt << " s elapsed    "
      << std::setw(7) << 1 / percent * dt - dt << " s remaining";
  return oss.str();
}

}