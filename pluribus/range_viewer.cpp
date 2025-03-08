#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include <utility>
#include <SDL2/SDL_image.h>
#include <pluribus/util.hpp>
#include <pluribus/actions.hpp>
#include <pluribus/range_viewer.hpp>

namespace pluribus {

Color Color::DARK_RED{169, 74, 61, 255};
Color Color::RED{193, 106, 87, 255};
Color Color::LIGHT_RED{212, 130, 107, 255};
Color Color::YELLOW{253, 254, 2, 255};
Color Color::DARK_GREEN{43, 179, 85, 255};
Color Color::GREEN{143, 189, 139, 255};
Color Color::BLUE{108, 162, 193, 255};
Color Color::CYAN{21, 235, 199, 255};
Color Color::PINK{235, 21, 221, 255};
Color Color::PURPLE{171, 21, 235, 255};


std::unordered_map<Action, Color> map_colors(const std::vector<Action>& actions) {
  const std::vector<Color> color_order{Color::DARK_RED, Color::RED, Color::LIGHT_RED, Color::YELLOW, Color::CYAN, 
    Color::PINK, Color::PURPLE, Color::DARK_GREEN};
  const std::unordered_map<Action, Color> special_colors{{Action::FOLD, Color::BLUE}, {Action::CHECK_CALL, Color::GREEN}};

  std::unordered_map<Action, Color> color_map;
  std::vector<Action> normal_actions;
  for(Action a : actions) {
    if(a.get_bet_type() > 0) {
      normal_actions.push_back(a);
    }
  }
  std::sort(normal_actions.begin(), normal_actions.end(), [](const auto& a, const auto& b) { return a.get_bet_type() > b.get_bet_type(); });
  if(std::find(actions.begin(), actions.end(), Action::ALL_IN) != actions.end()) {
    normal_actions.insert(normal_actions.begin(), Action::ALL_IN);
  }
  if(normal_actions.size() > color_order.size()) {
    throw std::runtime_error("Can map at most " + std::to_string(color_order.size()) + " actions to colors. " +
                             "n_actions=" + std::to_string(normal_actions.size()));
  }

  for(int i = 0; i < normal_actions.size(); ++i) {
    color_map[normal_actions[i]] = color_order[i];
  }
  for(const auto& entry : special_colors) {
    if(std::find(actions.begin(), actions.end(), entry.first) != actions.end()) {
      color_map[entry.first] = entry.second;
    }
  }
  if(actions.size() != color_map.size()) {
    for(const auto& entry : color_map) std::cout << "Found: " << entry.first.to_string() << "\n";
    throw std::runtime_error("Failed to map some colors: actions=" + std::to_string(actions.size()) +
                             ", mapped=" + std::to_string(color_map.size()));
  }
  return color_map;
}

RenderableRange::RenderableRange(const PokerRange& range, const Color& color, bool relative) 
    : _range{range}, _color{color.sdl_color}, _relative{relative} {
  int row_idx, col_idx, combos;
  for(const auto& combo : _range.range()) {
    if(combo.first.cards()[0] % 4 == combo.first.cards()[1] % 4) {
      row_idx = 0;
      col_idx = 1;
      combos = 4;
    }
    else {
      row_idx = 1;
      col_idx = 0;
      combos = combo.first.cards()[0] / 4 == combo.first.cards()[1] / 4 ? 6 : 12;
    }
    int row = _matrix.size() - combo.first.cards()[row_idx] / 4 - 1;
    int col = _matrix.size() - combo.first.cards()[col_idx] / 4 - 1;
    _matrix[row][col] += combo.second / combos;
  }
}

RangeViewer::RangeViewer(const std::string& title, int width, int height)
    : _window_width(width), _window_height(height), _title(title) {
  if(SDL_Init(SDL_INIT_VIDEO) < 0 || TTF_Init() < 0) {
    SDL_Log("Initialization failed: %s", SDL_GetError());
    return;
  }

  std::string font_path = std::string(PROJECT_ROOT_DIR) + "/resources/UbuntuMono-Regular.ttf";
  _font = TTF_OpenFont(font_path.c_str(), 24);
  if(!_font) {
    SDL_Log("Failed to load font: %s", TTF_GetError());
  }
}

RangeViewer::~RangeViewer() {
  if(_font) TTF_CloseFont(_font);
  IMG_Quit();
  TTF_Quit();
  SDL_Quit();
}

void RangeViewer::render_ranges(const std::vector<RenderableRange>& ranges) {
  const RenderableRange* base_rng = nullptr;
  for(const auto& rng : ranges) {
    if(!rng.is_relative()) {
      if(!base_rng) {
        base_rng = &rng;
      }
      else {
        throw std::runtime_error("RangeViewer --- Multiple absolute ranges given.");
      }
    }
  }
  SDL_SetRenderDrawColor(get_renderer(), 255, 255, 255, 255);
  SDL_RenderClear(get_renderer());
  render_background();
  RangeMatrix<int> cum_matrix;
  for(const auto& rng : ranges) render_range(&rng, &rng != base_rng ? base_rng : nullptr, cum_matrix);
  render_overlay();
}

void set_color(SDL_Renderer* renderer, const SDL_Color& color) {
  SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
}

SDL_Rect RangeViewer::make_rect(int row, int col, float freq, float rel_freq, int offset) const {
  int rect_h = static_cast<int>(round(_field_sz * freq));
  SDL_Rect rect = {
    static_cast<int>(round(_margin_x + col * _field_sz + offset)),
    static_cast<int>(round(_margin_y + row * _field_sz + _field_sz - rect_h)),
    static_cast<int>(round(_field_sz * rel_freq)), 
    rect_h
  };
  return rect;
}

void RangeViewer::render_hand(const SDL_Color& color, int row, int col, float freq, float rel_freq, int offset) {
  if(freq > 1.0f || rel_freq > 1.0f) {
    std::runtime_error("RangeViewer --- frequency overflow. freq=" + std::to_string(freq) + ", rel_freq" + std::to_string(rel_freq));
  }
  SDL_Rect action_rect = make_rect(row, col, freq, rel_freq, offset);
  set_color(get_renderer(), color);
  SDL_RenderFillRect(get_renderer(), &action_rect);
}

void RangeViewer::render_background() {
  for(int row = 0; row < 13; ++row) {
    for(int col = 0; col < 13; ++col) {
      render_hand(_field_bg_color, row, col, 1.0f);
    }
  } 
}

void RangeViewer::render_range(const RenderableRange* range, const RenderableRange* base_range, 
                               RangeMatrix<int>& offset_matrix) {
  for(int row = 0; row < range->get_matrix().size(); ++row) {
    for(int col = 0; col < range->get_matrix()[row].size(); ++col) {
      float freq = !base_range ? range->get_matrix()[row][col] : base_range->get_matrix()[row][col];
      float rel_freq = !base_range ? 1.0f : range->get_matrix()[row][col] / base_range->get_matrix()[row][col];
      render_hand(range->get_color(), row, col, freq, rel_freq, offset_matrix[row][col]);
      if(base_range) offset_matrix[row][col] += static_cast<int>(round(_field_sz * rel_freq));
    }
  }
}

void RangeViewer::render_overlay() {
  for(int row = 0; row < 13; ++row) {
    for(int col = 0; col < 13; ++col) {
      SDL_Rect border_rect = make_rect(row, col, 1.0f, 1.0f, 0.0f);
      set_color(get_renderer(), {0, 0, 0, 255});
      SDL_RenderDrawRect(get_renderer(), &border_rect);

      int major = col < row ? col : row;
      int minor = col < row ? row : col;
      std::string suit = row == col ? "" : (row > col ? "o" : "s");
      std::string label = std::string(1, omp::RANKS[omp::RANKS.size() - major - 1]) + omp::RANKS[omp::RANKS.size() - minor - 1] + suit;
      SDL_Surface* text_surface = TTF_RenderText_Solid(_font, label.c_str(), _text_color);
      SDL_Texture* text_texture = SDL_CreateTextureFromSurface(get_renderer(), text_surface);
      SDL_Rect text_rect = {border_rect.x + 5, border_rect.y + 3, text_surface->w, text_surface->h};
      SDL_RenderCopy(get_renderer(), text_texture, nullptr, &text_rect);
      SDL_FreeSurface(text_surface);
      SDL_DestroyTexture(text_texture);
    }
  } 
}

WindowRangeViewer::WindowRangeViewer(const std::string& title, int width, int height) : RangeViewer(title, width, height) {
  _window = SDL_CreateWindow(_title.c_str(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 
                            _window_width, _window_height, SDL_WINDOW_SHOWN);
  if(!_window) {
    SDL_Log("SDL_CreateWindow failed: %s", SDL_GetError());
    return;
  }
  _window_renderer = SDL_CreateRenderer(_window, -1, SDL_RENDERER_ACCELERATED);
  if(!_window_renderer) {
    SDL_Log("SDL_CreateRenderer failed: %s", SDL_GetError());
    return;
  }
}

WindowRangeViewer::~WindowRangeViewer() {
  if(_window_renderer) SDL_DestroyRenderer(_window_renderer);
  if(_window) SDL_DestroyWindow(_window);
}

void WindowRangeViewer::render(const std::vector<RenderableRange>& ranges) {
  render_ranges(ranges);
  SDL_RenderPresent(get_renderer());
}

PngRangeViewer::PngRangeViewer(const std::string& fn, int width, int height) : RangeViewer(fn, width, height), _fn{fn} {
  int imgFlags = IMG_INIT_PNG;
  if(!(IMG_Init(imgFlags) & imgFlags)) {
    SDL_Log("IMG_Init failed: %s", IMG_GetError());
    return;
  }
  _surface = SDL_CreateRGBSurface(0, _window_width, _window_height, 32, 0, 0, 0, 0);
  if(!_surface) {
    SDL_Log("SDL_CreateRGBSurface failed: %s", SDL_GetError());
    return;
  }
  _png_renderer = SDL_CreateSoftwareRenderer(_surface);
  if(!_png_renderer) {
    SDL_Log("SDL_CreateSoftwareRenderer failed: %s", SDL_GetError());
    return;
  }
  std::string font_path = std::string(PROJECT_ROOT_DIR) + "/resources/UbuntuMono-Regular.ttf";
  _font = TTF_OpenFont(font_path.c_str(), 24);
  if(!_font) {
    SDL_Log("Failed to load font: %s", TTF_GetError());
  }
}

PngRangeViewer::~PngRangeViewer() {
  if(_png_renderer) SDL_DestroyRenderer(_png_renderer);
  if(_surface) SDL_FreeSurface(_surface);
}

void PngRangeViewer::render(const std::vector<RenderableRange>& ranges) {
  render_ranges(ranges);
  if(IMG_SavePNG(_surface, _fn.c_str()) < 0) {
    SDL_Log("IMG_SavePNG failed: %s", IMG_GetError());
  } 
  else {
    SDL_Log("Saved to %s", _fn.c_str());
  }
}

}