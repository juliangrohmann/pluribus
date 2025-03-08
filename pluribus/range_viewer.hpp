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
  static Color DARK_GREEN;
  static Color GREEN;
  static Color BLUE;
  static Color YELLOW;
  static Color CYAN;
  static Color PINK;
  static Color PURPLE;
};

std::unordered_map<Action, Color> map_colors(const std::vector<Action>& actions);

template <class T>
class RangeMatrix {
public:
  RangeMatrix() : _matrix{13} { for(int row = 0; row < 13; ++row) _matrix[row].resize(13); }
  const std::vector<T>& operator[](int row) const { return _matrix[row]; }
  std::vector<T>& operator[](int row) { return _matrix[row]; }
  void clear() { for(int row = 0; row < 13; ++row) for(int col = 0; col < 13; ++col) _matrix[row][col] = 0.0f; }
  size_t size() const { return _matrix.size(); }
private:
  std::vector<std::vector<T>> _matrix;
};

class RenderableRange {
public:
  RenderableRange(const PokerRange& range, const Color& color, bool relative = false);

  const PokerRange& get_range() const { return _range; }
  const SDL_Color& get_color() const { return _color; }
  bool is_relative() const { return _relative; }
  const RangeMatrix<float>& get_matrix() const { return _matrix; }

private:
  PokerRange _range;
  SDL_Color _color;
  bool _relative = false;
  RangeMatrix<float> _matrix;
};

class RangeViewer {
public:
  RangeViewer(const std::string& title, int width = 1300, int height = 1300);
  ~RangeViewer();

  void render(const RenderableRange& range) { render({range}); }
  virtual void render(const std::vector<RenderableRange>& ranges) = 0;

protected:
  void render_ranges(const std::vector<RenderableRange>& ranges);
  SDL_Rect make_rect(int row, int col, float freq, float rel_freq, int offset) const;
  void render_hand(const SDL_Color& color, int row, int col, float freq, float rel_freq = 1.0f, int offset = 0);
  void render_background();
  void render_range(const RenderableRange* range, const RenderableRange* base_range, RangeMatrix<int>& offset_matrix);
  void render_overlay();
  virtual SDL_Renderer* get_renderer() = 0;

  int _window_width, _window_height;
  std::string _title;

  TTF_Font* _font = nullptr;

  int _margin_x = 0;
  int _margin_y = 0;
  int _field_sz = 100;
  SDL_Color _text_color{0, 0, 0, 255};
  SDL_Color _field_bg_color{103, 103, 103, 255};
};

class WindowRangeViewer : public RangeViewer {
public:
  WindowRangeViewer(const std::string& title, int width = 1300, int height = 1300);
  ~WindowRangeViewer();

  void render(const std::vector<RenderableRange>& ranges) override;

private:
  SDL_Renderer* get_renderer() override { return _window_renderer; }

  SDL_Renderer* _window_renderer = nullptr;
  SDL_Window* _window = nullptr;
};

class PngRangeViewer : public RangeViewer {
public:
  PngRangeViewer(const std::string& fn, int width = 1300, int height = 1300);
  ~PngRangeViewer();

  void render(const std::vector<RenderableRange>& ranges) override;

private:
  SDL_Renderer* get_renderer() override { return _png_renderer; }

  std::string _fn;
  SDL_Renderer* _png_renderer = nullptr;
  SDL_Surface* _surface = nullptr;
};

}
