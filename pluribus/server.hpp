#pragma once

#include <vector>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <atomic>
#include <memory>
#include <httplib/httplib.h>
#include <pluribus/pluribus.hpp>
#include <pluribus/poker.hpp>

namespace pluribus {

enum class CommandType { NewGame, UpdateState, UpdateBoard };

struct Command {
  CommandType type;
  std::vector<std::string> players;
  std::vector<uint8_t> board;
  Action action;
  int pos;
};

class PluribusServer {
public:
  PluribusServer(const std::string& preflop_fn, const std::string& sampled_fn);
  ~PluribusServer() { stop(); }

  void start();

private:
  void stop();
  void dispatch_commands();

  static void receive_commands();
  void configure_server();

  std::unique_ptr<Pluribus> _engine;
  std::shared_ptr<const LosslessBlueprint> _preflop_bp;
  std::shared_ptr<const SampledBlueprint> _sampled_bp;

  httplib::Server _server;
  std::thread _dispatch_thread;
  std::mutex _cmd_mtx;
  std::condition_variable _cmd_cv;
  std::deque<Command> _cmd_queue;
  std::atomic<bool> _running;
};

}
