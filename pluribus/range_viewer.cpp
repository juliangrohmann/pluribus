#include <algorithm>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <pluribus/actions.hpp>
#include <pluribus/range_viewer.hpp>
#include <pluribus/util.hpp>
#include <SDL2/SDL_image.h>

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
  const std::vector color_order{Color::DARK_RED, Color::RED, Color::LIGHT_RED, Color::YELLOW, Color::CYAN,
    Color::PINK, Color::PURPLE, Color::DARK_GREEN};
  const std::unordered_map<Action, Color> special_colors{{Action::FOLD, Color::BLUE}, {Action::CHECK_CALL, Color::GREEN}};

  std::unordered_map<Action, Color> color_map;
  std::vector<Action> normal_actions;
  for(Action a : actions) {
    if(a.get_bet_type() > 0) {
      normal_actions.push_back(a);
    }
  }
  std::ranges::sort(normal_actions, [](const auto& a, const auto& b) { return a.get_bet_type() > b.get_bet_type(); });
  if(std::ranges::find(actions, Action::ALL_IN) != actions.end()) {
    normal_actions.insert(normal_actions.begin(), Action::ALL_IN);
  }
  if(normal_actions.size() > color_order.size()) {
    throw std::runtime_error("Can map at most " + std::to_string(color_order.size()) + " actions to colors. " +
                             "n_actions=" + std::to_string(normal_actions.size()));
  }

  for(int i = 0; i < normal_actions.size(); ++i) {
    color_map[normal_actions[i]] = color_order[i];
  }
  for(const auto&[a, color] : special_colors) {
    if(std::ranges::find(actions, a) != actions.end()) {
      color_map[a] = color;
    }
  }
  if(actions.size() != color_map.size()) {
    for(const auto& action : color_map | std::views::keys) std::cout << "Found: " << action.to_string() << "\n";
    throw std::runtime_error("Failed to map some colors: actions=" + std::to_string(actions.size()) +
                             ", mapped=" + std::to_string(color_map.size()));
  }
  return color_map;
}

RenderableRange::RenderableRange(const PokerRange& range, const std::string& label, const Color& color, const bool relative)
    : _range{range}, _label{label}, _color{color.sdl_color}, _relative{relative} {
  int row_idx, col_idx, combos;
  const auto& weights = _range.weights();
  for(int idx = 0; idx < weights.size(); ++idx) {
    Hand hand = HoleCardIndexer::get_instance()->hand(idx);
    if(hand.cards()[0] % 4 == hand.cards()[1] % 4) {
      row_idx = 0;
      col_idx = 1;
      combos = 4;
    }
    else {
      row_idx = 1;
      col_idx = 0;
      combos = hand.cards()[0] / 4 == hand.cards()[1] / 4 ? 6 : 12;
    }
    const int row = _matrix.size() - hand.cards()[row_idx] / 4 - 1;
    const int col = _matrix.size() - hand.cards()[col_idx] / 4 - 1;
    _matrix[row][col] += weights[idx] / combos;
  }
}

RangeViewer::RangeViewer(const std::string& title, const int width, const int height)
    : _window_width(width), _window_height(height), _title(title) {
  if(SDL_Init(SDL_INIT_VIDEO) < 0 || TTF_Init() < 0) {
    SDL_Log("Initialization failed: %s", SDL_GetError());
    return;
  }

  const std::string font_path = std::string(PROJECT_ROOT_DIR) + "/resources/UbuntuMono-Regular.ttf";
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
  render_legend(ranges);
  render_overlay();
}

void set_color(SDL_Renderer* renderer, const SDL_Color& color) {
  SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
}

void RangeViewer::render_text(const std::string& label, const int x, const int y) {
  SDL_Surface* text_surface = TTF_RenderText_Solid(_font, label.c_str(), _text_color);
  SDL_Texture* text_texture = SDL_CreateTextureFromSurface(get_renderer(), text_surface);
  const SDL_Rect text_rect = {x, y, text_surface->w, text_surface->h};
  SDL_RenderCopy(get_renderer(), text_texture, nullptr, &text_rect);
  SDL_FreeSurface(text_surface);
  SDL_DestroyTexture(text_texture);
}

SDL_Rect RangeViewer::make_rect(const int row, const int col, const float freq, const float rel_freq, const int offset) const {
  const int rect_h = static_cast<int>(round(_field_sz * freq));
  const SDL_Rect rect = {
    static_cast<int>(round(_margin_x + col * _field_sz + offset)),
    static_cast<int>(round(_margin_y + row * _field_sz + _field_sz - rect_h)),
    static_cast<int>(round(_field_sz * rel_freq)), 
    rect_h
  };
  return rect;
}

void RangeViewer::render_hand(const SDL_Color& color, const int row, const int col, const float freq, const float rel_freq, const int offset) {
  if(freq > 1.0f || rel_freq > 1.0f) {
    throw std::runtime_error("RangeViewer --- frequency overflow. freq=" + std::to_string(freq) + ", rel_freq" + std::to_string(rel_freq));
  }
  const SDL_Rect action_rect = make_rect(row, col, freq, rel_freq, offset);
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
      const float freq = !base_range ? range->get_matrix()[row][col] : base_range->get_matrix()[row][col];
      const float rel_freq = !base_range ? 1.0f : range->get_matrix()[row][col] / base_range->get_matrix()[row][col];
      render_hand(range->get_color(), row, col, freq, rel_freq, offset_matrix[row][col]);
      if(base_range) offset_matrix[row][col] += static_cast<int>(round(_field_sz * rel_freq));
      
      if(row == 0 && col == 2 && base_range) {
        std::cout << "action=" << range->get_label() << ", freq=" << freq << ", rel_freq=" << rel_freq << ", matrix=" << range->get_matrix()[row][col] << ", base_matrix=" << base_range->get_matrix()[row][col] << "\n";
      }
    }
  }
}

void RangeViewer::render_legend(const std::vector<RenderableRange>& ranges) {
  int n_rel_ranges = 0;
  float n_combos = 0;
  for(const auto& range : ranges) {
    if(range.is_relative()) {
      n_combos += range.get_range().n_combos();
      ++n_rel_ranges;
    }
  }
  if(n_rel_ranges == 0) return;
  
  const int w = _window_width / n_rel_ranges;
  int x = 0;
  for(const auto& range : ranges) {
    if(range.is_relative()) {
      constexpr int h = 100;
      SDL_Rect rect = {x, _window_height - h, w, h};
      set_color(get_renderer(), range.get_color());
      SDL_RenderFillRect(get_renderer(), &rect);
      x += w;
      render_text(range.get_label(), rect.x + 5, rect.y + 3);
      std::ostringstream combos_oss;
      combos_oss << std::fixed << std::setprecision(1) << range.get_range().n_combos() << " combos";
      render_text(combos_oss.str(), rect.x + 5, rect.y + 23);
      std::ostringstream perc_oss;
      perc_oss << std::fixed << std::setprecision(2) << range.get_range().n_combos() * 100.0 / n_combos << "%";
      render_text(perc_oss.str(), rect.x + 5, rect.y + 43);
    }
  }
}

void RangeViewer::render_overlay() {
  for(int row = 0; row < 13; ++row) {
    for(int col = 0; col < 13; ++col) {
      SDL_Rect border_rect = make_rect(row, col, 1.0f, 1.0f, 0.0f);
      set_color(get_renderer(), {0, 0, 0, 255});
      SDL_RenderDrawRect(get_renderer(), &border_rect);

      const int major = col < row ? col : row;
      const int minor = col < row ? row : col;
      std::string suit = row == col ? "" : row > col ? "o" : "s";
      std::string label = std::string(1, omp::RANKS[omp::RANKS.size() - major - 1]) + omp::RANKS[omp::RANKS.size() - minor - 1] + suit;
      render_text(label, border_rect.x + 5, border_rect.y + 3);
    }
  } 
}

WindowRangeViewer::WindowRangeViewer(const std::string& title, const int width, const int height) : RangeViewer(title, width, height) {
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

PngRangeViewer::PngRangeViewer(const std::string& fn, const int width, const int height) : RangeViewer(fn, width, height), _fn{fn} {
  if(constexpr int imgFlags = IMG_INIT_PNG; !(IMG_Init(imgFlags) & imgFlags)) {
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
  const std::string font_path = std::string(PROJECT_ROOT_DIR) + "/resources/UbuntuMono-Regular.ttf";
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