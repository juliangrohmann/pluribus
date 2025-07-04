#include <json/json.hpp>
#include <pluribus/logging.hpp>
#include <pluribus/server.hpp>

using json = nlohmann::json;

namespace pluribus {
PluribusServer::PluribusServer(const std::string& preflop_fn, const std::string& sampled_fn)
    : _preflop_bp{std::make_shared<LosslessBlueprint>()}, _sampled_bp{std::make_shared<SampledBlueprint>()} {
  cereal_load(*_preflop_bp, preflop_fn);
  cereal_load(*_sampled_bp, sampled_fn);
  _engine = std::make_unique{_preflop_bp, _sampled_bp};
  configure_server();
}



void PluribusServer::start() {
  _dispatch_thread = std::thread{[this] { this->dispatch_commands(); }};
  receive_commands();
  stop();
}

void PluribusServer::stop() {
  _running = false;
  _cmd_cv.notify_one();
  _dispatch_thread.join();
}

void PluribusServer::dispatch_commands() {
  while(_running.load(std::memory_order_acquire)) {
    Command cmd;
    {
      std::unique_lock lock(_cmd_mtx);
      _cmd_cv.wait(lock, [&]{ return !_cmd_queue.empty() || !_running.load(); });
      if (!_running.load()) break;
      cmd = std::move(_cmd_queue.front());
      _cmd_queue.pop_front();
    }

    switch (cmd.type) {
      case CommandType::NewGame:
        _engine->new_game(cmd.players);
        break;
      case CommandType::UpdateState:
        _engine->update_state(cmd.action, cmd.pos);
        break;
      case CommandType::UpdateBoard:
        _engine->update_board(cmd.board);
        break;
      default:
        Logger::error("Failed to dispatch. Unknown command.");
    }
  }
}

void PluribusServer::receive_commands() {

}

void PluribusServer::configure_server() {
  _server.Post("/new_game", [&](auto& req, auto& res){
    // ReSharper disable once CppDeprecatedEntity
    auto dat = json::parse(req.body.begin(), req.body.end());
    const auto players = dat.at("players").template get<std::vector<std::string>>();
    Logger::log("POST: /new_game players=[" + join_strs(players, ", ") + "]");
    {
      std::lock_guard lock(_cmd_mtx);
      _cmd_queue.push_back(Command{CommandType::NewGame, players, {}, Action::UNDEFINED, -1});
    }
    _cmd_cv.notify_one();
    res.set_content(R"({"status":"ok"})", "application/json");
  });

  _server.Post("/update_state", [&](auto& req, auto& res){
    // ReSharper disable once CppDeprecatedEntity
    auto dat = json::parse(req.body.begin(), req.body.end());
    const auto action = Action{dat.at("action").template get<float>()};
    const auto pos = dat.at("pos").template get<int>();
    Logger::log("POST: /update_state action=" + action.to_string() + ", pos=" + pos);
    {
      std::lock_guard lock(_cmd_mtx);
      _cmd_queue.push_back(Command{CommandType::UpdateState, {}, {}, action, pos});
    }
    _cmd_cv.notify_one();
    res.set_content(R"({"status":"ok"})", "application/json");
  });

  _server.Post("/update_board", [&](auto& req, auto& res){
    // ReSharper disable once CppDeprecatedEntity
    auto dat = json::parse(req.body.begin(), req.body.end());
    auto board_str = j.at("board").get<std::string>();
    Logger::log("POST: /update_board board=" + board_str);
    {
      std::lock_guard lock(_cmd_mtx);
      _cmd_queue.push_back(Command{CommandType::UpdateBoard, {}, str_to_cards(board_str), Action::UNDEFINED, -1});
    }
    _cmd_cv.notify_one();
    res.set_content(R"({"status":"ok"})", "application/json");
  });

  _server.Get("/solution", [&](auto&, auto& res){
    // ReSharper disable once CppDeprecatedEntity
    auto dat = json::parse(req.body.begin(), req.body.end());
    if (!sol_ptr) {
      res.status = 503;
      res.set_content(R"({"error":"not ready"})", "application/json");
      return;
    }
    json j;
    j["actions"] = sol_ptr->actions;  // you'll need to implement to_json(Action)
    j["freq"]    = sol_ptr->freq;
    res.set_content(j.dump(), "application/json");
  });

}
}
