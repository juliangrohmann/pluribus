#include <json/json.hpp>
#include <pluribus/logging.hpp>
#include <pluribus/server.hpp>

using json = nlohmann::json;

namespace pluribus {
PluribusServer::PluribusServer(const std::string& preflop_fn, const std::string& sampled_fn)
    : _preflop_bp{std::make_shared<LosslessBlueprint>()}, _sampled_bp{std::make_shared<SampledBlueprint>()} {
  cereal_load(*_preflop_bp, preflop_fn);
  cereal_load(*_sampled_bp, sampled_fn);
  const int n_players = _preflop_bp->get_config().poker.n_players;
  std::array<ActionProfile, 4> live_profiles;
  for(int r = 0; r < 4; ++r) {
    // TODO: unify blueprint/live profiles into one class and read live profiles from _sampled_bp
    if(n_players > 2) live_profiles[r] = RingLiveProfile{n_players, r};
    else live_profiles[r] = HeadsUpLiveProfile{};
  }

  _engine = std::make_unique<Pluribus>(live_profiles, _preflop_bp, _sampled_bp);
  configure_server();
}

void PluribusServer::start() {
  Logger::log("Starting HTTP server on port 8080...");
  _running = true;
  _dispatch_thread = std::thread{[this] { this->dispatch_commands(); }};
  Logger::log("Listening...");
  _server.listen("0.0.0.0", 8080);
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
        _engine->new_game(cmd.stacks, cmd.hand, cmd.pos);
        break;
      case CommandType::UpdateState:
        _engine->update_state(cmd.action, cmd.pos);
        break;
      case CommandType::HeroAction:
        _engine->hero_action(cmd.action, cmd.freq);
        break;
      case CommandType::UpdateBoard:
        _engine->update_board(cmd.board);
        break;
      case CommandType::SaveRange:
        _engine->save_range(cmd.fn);
        break;
      default:
        Logger::error("Failed to dispatch. Unknown command.");
    }
  }
}

void PluribusServer::configure_server() {
  _server.Post("/new_game", [&](auto& req, auto& res){
    // ReSharper disable once CppDeprecatedEntity
    auto dat = json::parse(req.body.begin(), req.body.end());
    const auto stacks = dat.at("stacks").template get<std::vector<int>>();
    const auto hero_hand = Hand{dat.at("hero_hand").template get<std::string>()};
    const auto hero_pos = dat.at("hero_pos").template get<int>();
    Logger::log("POST: /new_game stacks=[" + join_as_strs(stacks, ", ") + "], hero_hand=" + hero_hand.to_string() + ", hero_pos=" + std::to_string(hero_pos));
    {
      std::lock_guard lock(_cmd_mtx);
      _cmd_queue.push_back(Command::make_new_game(stacks, hero_pos));
    }
    _cmd_cv.notify_one();
    res.set_content(R"({"status":"ok"})", "application/json");
  });

  _server.Post("/update_state", [&](auto& req, auto& res){
    // ReSharper disable once CppDeprecatedEntity
    auto dat = json::parse(req.body.begin(), req.body.end());
    const auto action = Action{dat.at("action").template get<float>()};
    const auto pos = dat.at("pos").template get<int>();
    Logger::log("POST: /update_state action=" + action.to_string() + ", pos=" + std::to_string(pos));
    {
      std::lock_guard lock(_cmd_mtx);
      _cmd_queue.push_back(Command::make_update_state(action, pos));
    }
    _cmd_cv.notify_one();
    res.set_content(R"({"status":"ok"})", "application/json");
  });

  _server.Post("/hero_action", [&](auto& req, auto& res){
    // ReSharper disable once CppDeprecatedEntity
    auto dat = json::parse(req.body.begin(), req.body.end());
    const auto action = Action{dat.at("action").template get<float>()};
    const auto freq = dat.at("freq").template get<std::vector<float>>();
    Logger::log("POST: /update_state action=" + action.to_string() + ", freq=[" + join_as_strs(freq, ", ") + "]");
    {
      std::lock_guard lock(_cmd_mtx);
      _cmd_queue.push_back(Command::make_hero_action(action, freq));
    }
    _cmd_cv.notify_one();
    res.set_content(R"({"status":"ok"})", "application/json");
  });

  _server.Post("/update_board", [&](auto& req, auto& res){
    // ReSharper disable once CppDeprecatedEntity
    auto dat = json::parse(req.body.begin(), req.body.end());
    auto board_str = dat.at("board").template get<std::string>();
    Logger::log("POST: /update_board board=" + board_str);
    {
      std::lock_guard lock(_cmd_mtx);
      _cmd_queue.push_back(Command::make_update_board(str_to_cards(board_str)));
    }
    _cmd_cv.notify_one();
    res.set_content(R"({"status":"ok"})", "application/json");
  });

  _server.Post("/solution", [&](auto& req, auto& res){
    // ReSharper disable once CppDeprecatedEntity
    auto dat = json::parse(req.body.begin(), req.body.end());
    const auto hand = Hand{dat.at("hand").template get<std::string>()};
    auto [actions, freq] = _engine->solution(hand);
    std::vector<std::string> str_actions;
    std::ranges::transform(actions.begin(), actions.end(), std::back_inserter(str_actions), [](const Action a) { return a.to_string(); });
    json j;
    j["actions"] = str_actions;
    j["freq"]    = freq;
    j["status"]  = "ok";
    res.set_content(j.dump(), "application/json");
  });

  _server.Post("/save_range", [&](auto& req, auto& res){
    // ReSharper disable once CppDeprecatedEntity
    auto dat = json::parse(req.body.begin(), req.body.end());
    const auto fn = dat.at("fn").template get<std::string>();
    Logger::log("POST: /save_range fn=" + fn);
    {
      std::lock_guard lock(_cmd_mtx);
      _cmd_queue.push_back(Command::make_save_range(fn));
    }
    _cmd_cv.notify_one();
    res.set_content(R"({"status":"ok"})", "application/json");
  });

}
}
