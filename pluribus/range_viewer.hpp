#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <pluribus/range.hpp>

namespace pluribus {

struct Square {
    int x, y, size;
    Uint8 r, g, b;
    std::string label;
};

struct Color {
  SDL_Color sdl_color;

  static Color DARK_RED;
  static Color RED;
  static Color LIGHT_RED;
  static Color YELLOW;
  static Color DARK_GREEN;
  static Color GREEN;
  static Color BLUE;
};

class RangeMatrix {
public:
  RangeMatrix() : _matrix{13} { for(int row = 0; row < 13; ++row) _matrix[row].resize(13); }
  const std::vector<float>& operator[](int row) const { return _matrix[row]; }
  std::vector<float>& operator[](int row) { return _matrix[row]; }
  void clear() { for(int row = 0; row < 13; ++row) for(int col = 0; col < 13; ++col) _matrix[row][col] = 0.0f; }
  size_t size() const { return _matrix.size(); }
private:
  std::vector<std::vector<float>> _matrix;
};

class RenderableRange {
public:
  RenderableRange(const PokerRange& range, const Color& color, bool relative = false);

  const PokerRange& get_range() const { return _range; }
  const SDL_Color& get_color() const { return _color; }
  bool is_relative() const { return _relative; }
  const RangeMatrix& get_matrix() const { return _matrix; }

private:
  PokerRange _range;
  SDL_Color _color;
  bool _relative = false;
  RangeMatrix _matrix;
};

class RangeViewer {
public:
  RangeViewer(const std::string& title, int width = 1300, int height = 1300);
  ~RangeViewer();

  void render(const RenderableRange& range);
  void render(const std::vector<RenderableRange>& range);

private:
  SDL_Rect make_rect(int row, int col, float freq, float rel_freq, float cum_freq) const;
  void render_hand(const SDL_Color& color, int row, int col, float freq, float rel_freq = 1.0f, float cum_freq = 0.0f);
  void render_background();
  void render_range(const RenderableRange* range, const RenderableRange* base_range, RangeMatrix& cum_matrix);
  void render_overlay();
  
  int _window_width, _window_height;
  std::string _title;
  SDL_Window* _window = nullptr;
  SDL_Renderer* _renderer = nullptr;
  TTF_Font* _font = nullptr;
  std::vector<Square> squares;
  int _margin_x = 0;
  int _margin_y = 0;
  int _field_sz = 100;
  SDL_Color _text_color{0, 0, 0, 255};
  SDL_Color _field_bg_color{103, 103, 103, 255};
};

}
