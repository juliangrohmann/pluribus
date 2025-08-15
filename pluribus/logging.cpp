#include <pluribus/logging.hpp>

namespace pluribus {

std::unique_ptr<Logger> Logger::_instance = nullptr;

std::string progress_str(const hand_index_t idx, const hand_index_t total, const std::chrono::high_resolution_clock::time_point& t_0) {
  const auto dt = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - t_0).count();
  const double percent = static_cast<double>(idx) / static_cast<double>(total);
  std::ostringstream oss;
  oss << std::setw(11) << std::to_string(idx) << ":   "
      << std::fixed << std::setprecision(1) << std::setw(5) << (percent * 100) << "%"
      << "    Elapsed: " << std::setw(7) << std::setprecision(0) << dt << " s"
      << "    Remaining: " << std::setw(7) << 1 / percent * dt - dt << " s";
  return oss.str();
}

}