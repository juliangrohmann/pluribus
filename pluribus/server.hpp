#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>
#include <httplib/httplib.h>
#include <pluribus/pluribus.hpp>
#include <pluribus/poker.hpp>

namespace pluribus {

enum class CommandType { NewGame, UpdateState, UpdateBoard, Solution, SaveRange };

struct Command {
  CommandType type;
  std::vector<int> stacks{};
  std::vector<uint8_t> board{};
  Hand hand{};
  Action action = Action::UNDEFINED;
  int pos = -1;
  std::string fn;

  static Command make_new_game(const std::vector<int>& stacks_, int hero_pos) {
    return Command{CommandType::NewGame, stacks_, {}, {}, {}, Action::UNDEFINED, hero_pos};
  }
  static Command make_update_state(const Action action_, const int pos_) {
    return Command{CommandType::UpdateState, {}, {}, {}, action_, pos_};
  }
  static Command make_update_board(const std::vector<uint8_t>& board_) {
    return Command{CommandType::UpdateBoard, {}, board_};
  }
  static Command make_solution(const Hand& hand_) {
    return Command{CommandType::Solution, {}, {}, hand_};
  }
  static Command make_save_range(const std::string& fn_) {
    return Command{CommandType::SaveRange, {}, {}, {}, Action::UNDEFINED, -1, fn_};
  }
};

class PluribusServer {
public:
  PluribusServer(const std::string& preflop_fn, const std::string& sampled_fn);
  ~PluribusServer() { stop(); }

  void start();

private:
  void stop();
  void dispatch_commands();

  void configure_server();

  std::unique_ptr<Pluribus> _engine;
  std::shared_ptr<LosslessBlueprint> _preflop_bp;
  std::shared_ptr<SampledBlueprint> _sampled_bp;

  httplib::Server _server;
  std::thread _dispatch_thread;
  std::mutex _cmd_mtx;
  std::condition_variable _cmd_cv;
  std::deque<Command> _cmd_queue;
  std::atomic<bool> _running;
};

}
