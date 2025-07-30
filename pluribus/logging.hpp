#pragma once

#include <filesystem>
#include <iostream>
#include <memory>
#include <pluribus/util.hpp>

namespace pluribus {

class Log {
public:
  explicit Log(const std::filesystem::path& fn, const int debug = 0) : _fn{fn}, _debug{debug} {}

  void log(const std::string& msg, const int debug = 0) const {
    if(_debug >= debug) {
      if(const auto dir = _fn.parent_path(); !create_dir(dir)) throw std::runtime_error("Failed to create log directory \"" + dir.string() + "\"");
      const std::string line = date_time_str("%m/%d/%Y %H:%M:%S") + ": " + msg + "\n";
      std::cout << line;
      write_to_file(_fn, line, true);
    }
  }

  [[noreturn]] void error(const std::string& msg) const {
    log("Error: " + msg);
    throw std::runtime_error(msg);
  }

  void set_debug(const int debug) { _debug = debug; }

private:
  std::filesystem::path _fn;
  int _debug;
};

class Logger {
public:
  static void dump(std::ostringstream& buf) {
    instance()->_log.log(buf.str());
    buf.str("");
  }

  static void log(const std::string& msg, const int debug = 0) {
    instance()->_log.log(msg, debug);
  }

  [[noreturn]] static void error(const std::string& msg) {
    instance()->_log.error(msg);
  }

  static void set_directory(const std::filesystem::path &dir) {
    set_log(Log{dir / (date_time_str() + ".log")});
  }

  static void set_filename(const std::filesystem::path &fn) {
    set_log(Log{fn});
  }

  static void set_log(const Log& new_log) {
    instance()->_log = new_log;
  }

private:
  static Logger* instance() { 
    if(!_instance) {
      _instance = std::unique_ptr<Logger>{new Logger{}};
    }
    return _instance.get();
  }

  Logger() : _log{std::filesystem::path{"logs"} / (date_time_str() + ".log")} {}

  static std::unique_ptr<Logger> _instance;

  Log _log;
};

}