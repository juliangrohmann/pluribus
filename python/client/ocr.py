# pip install onnxruntime-gpu opencv-python numpy pillow
import os
import cv2
import numpy as np
import onnxruntime as ort
import argparse
import PIL.Image as Image
from typing import Tuple, List

charset_path = r"resources\en_dict.txt"

# ---------- utility: geometry ----------

def order_points_clockwise(pts: np.ndarray) -> np.ndarray:
  # pts: (4,2)
  rect = np.zeros((4, 2), dtype=np.float32)
  s = pts.sum(axis=1)
  rect[0] = pts[np.argmin(s)]  # top-left
  rect[2] = pts[np.argmax(s)]  # bottom-right
  diff = np.diff(pts, axis=1).ravel()
  rect[1] = pts[np.argmin(diff)]  # top-right
  rect[3] = pts[np.argmax(diff)]  # bottom-left
  return rect

import numpy as np

def refine_crop_with_prob_segments(x0:int, y0:int, x1:int, y1:int,
                                   prob_up: np.ndarray, sh: float, sw: float,
                                   margin_px: int = 3,
                                   gap_px_prob: int = 6,   # max gap in prob cols to merge runs
                                   smooth_w: int = 5,      # 1D smoothing over columns
                                   min_width_px: int = 12  # enforce a minimum width in orig px
                                   ) -> tuple[int,int,int,int]:
  # map crop → prob coords
  xs0 = max(0, int(round(x0 * sw)));  xs1 = min(prob_up.shape[1], int(round(x1 * sw)))
  ys0 = max(0, int(round(y0 * sh)));  ys1 = min(prob_up.shape[0], int(round(y1 * sh)))
  if xs1 <= xs0 or ys1 <= ys0:
    return x0, y0, x1, y1

  region = prob_up[ys0:ys1, xs0:xs1]            # (h,w) in prob space
  col = region.mean(axis=0).astype(np.float32)  # [w]

  # smooth to bridge tiny dips between '$' and digits
  if smooth_w > 1:
    k = np.ones(smooth_w, dtype=np.float32) / smooth_w
    col = np.convolve(col, k, mode='same')

  # adaptive threshold; keep more columns for short strings
  thr = max(np.percentile(col, 60) * 0.75, 0.25 * float(col.max()))
  mask = col >= thr
  if not mask.any():
    return x0, y0, x1, y1

  # find contiguous runs
  runs = []
  i = 0
  Wp = mask.size
  while i < Wp:
    if mask[i]:
      j = i
      while j < Wp and mask[j]:
        j += 1
      runs.append([i, j])  # [start, end)
      i = j
    else:
      i += 1

  # merge runs separated by small gaps (<= gap_px_prob columns in prob space)
  merged = []
  for s, e in runs:
    if not merged:
      merged.append([s, e])
    else:
      if s - merged[-1][1] <= gap_px_prob:
        merged[-1][1] = e
      else:
        merged.append([s, e])

  # score runs by total mass (integral), choose the best
  best = max(merged, key=lambda se: float(col[se[0]:se[1]].sum()))
  i0, i1 = best

  # map back to original pixels (+ margin) and enforce a minimum width
  nx0 = x0 + int(round(i0 / sw))
  nx1 = x0 + int(round(i1 / sw))
  if nx1 - nx0 < min_width_px:
    cx = (nx0 + nx1) // 2
    nx0 = max(0, cx - min_width_px // 2)
    nx1 = cx + (min_width_px - (cx - nx0))

  nx0 = max(0, nx0 - margin_px)
  nx1 = nx1 + margin_px
  return nx0, y0, nx1, y1

def quad_to_aabb(quad: np.ndarray, img_w: int, img_h: int, expand: int = 0) -> tuple[int,int,int,int]:
  """
  quad: (4,2) float32 in image coords
  returns (x0,y0,x1,y1) clipped to image, optionally expanded by 'expand' pixels.
  """
  xs = quad[:,0]; ys = quad[:,1]
  x0 = int(np.floor(xs.min())) - expand
  y0 = int(np.floor(ys.min())) - expand
  x1 = int(np.ceil(xs.max()))  + expand
  y1 = int(np.ceil(ys.max()))  + expand
  x0 = max(0, x0); y0 = max(0, y0)
  x1 = min(img_w, x1); y1 = min(img_h, y1)
  if x1 <= x0 or y1 <= y0:
    # empty after clipping → return full image as a safe fallback
    return 0, 0, img_w, img_h
  return x0, y0, x1, y1

def border_median_color(img_rgb: np.ndarray, k: int = 3) -> tuple[int,int,int]:
  """
  Estimate background color from a k-px border frame using per-channel median.
  Returns (R,G,B) uint8.
  """
  h, w = img_rgb.shape[:2]
  k = min(k, h // 2, w // 2) if min(h, w) > 0 else 0
  if k <= 0:
    return (255, 255, 255)
  top = img_rgb[:k, :, :]
  bot = img_rgb[-k:, :, :]
  lef = img_rgb[:, :k, :]
  rig = img_rgb[:, -k:, :]
  border = np.concatenate([top.reshape(-1,3), bot.reshape(-1,3),
                           lef.reshape(-1,3), rig.reshape(-1,3)], axis=0)
  med = np.median(border, axis=0).astype(np.uint8)
  return int(med[0]), int(med[1]), int(med[2])  # R,G,B

# ---------- detector (PP-OCRv3 DB) ----------

class PPOCRv3Detector:
  def __init__(self,
               onnx_path: str,
               device_id: int = 0,
               max_side: int = 960,
               bin_thresh: float = 0.3,
               box_thresh: float = 0.5,
               unclip_ratio: float = 1.6,
               use_trt: bool = True,
               cache_dir: str = "trt_cache"):
    self.max_side = max_side
    self.bin_thresh = bin_thresh
    self.box_thresh = box_thresh
    self.unclip_ratio = unclip_ratio

    providers = []
    trt_opts = {
      "device_id": device_id,
      "trt_fp16_enable": True,
      "trt_max_workspace_size": 1 << 30,
      "trt_engine_cache_enable": True,
      "trt_engine_cache_path": cache_dir,
      "trt_timing_cache_enable": True,
      "trt_builder_optimization_level": 5,
    }
    if use_trt:
      providers.append(("TensorrtExecutionProvider", trt_opts))
    providers.append(("CUDAExecutionProvider", {"device_id": device_id}))
    providers.append("CPUExecutionProvider")

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
      self.sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    except Exception:
      self.sess = ort.InferenceSession(onnx_path, sess_options=so,
        providers=[("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"])
    self.input_name = self.sess.get_inputs()[0].name
    self._warmup(sizes=[(64, 256), (96, 384), (128, 512)])

  def _warmup(self, sizes: list[tuple[int, int]] = [(96, 384)], iters: int = 2) -> None:
    """
    Run the full pipeline on blank BGR images to build TRT engines for common shapes.
    sizes are pre-pad HxW (your _resize_keep_ratio will pad to /32).
    """
    for (h, w) in sizes:
      dummy = np.zeros((h, w, 3), dtype=np.uint8)  # BGR
      for _ in range(iters):
        _ = self.infer_boxes(*self.calc_metrics(dummy))

  @staticmethod
  def _normalize_imagenet(img_bgr: np.ndarray) -> np.ndarray:
    img = img_bgr.astype(np.float32) / 255.0
    mean = np.array([0.406, 0.456, 0.485], dtype=np.float32)[None, None, :]
    std = np.array([0.225, 0.224, 0.229], dtype=np.float32)[None, None, :]
    img = (img - mean) / std
    return img

  @staticmethod
  def _pad_to_stride(h: int, w: int, stride: int = 32) -> Tuple[int, int]:
    return int(np.ceil(h / stride) * stride), int(np.ceil(w / stride) * stride)

  def _resize_keep_ratio(self, img_bgr: np.ndarray):
    h0, w0 = img_bgr.shape[:2]
    scale = self.max_side / max(h0, w0) if max(h0, w0) > self.max_side else 1.0
    nh, nw = int(round(h0 * scale)), int(round(w0 * scale))
    nh32, nw32 = int(np.ceil(nh / 32) * 32), int(np.ceil(nw / 32) * 32)
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad = np.full((nh32, nw32, 3), 255, dtype=np.uint8)
    pad[:nh, :nw] = resized
    # return padded image plus native resized dims and scales
    return pad, nh, nw, (nh / h0), (nw / w0)

  @staticmethod
  def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

  @staticmethod
  def _find_boxes_from_prob(prob_up, nh, nw, bin_thresh, box_thresh, dilate=(0, 0)):
    binmap = (prob_up >= bin_thresh).astype(np.uint8) * 255
    binmap = cv2.morphologyEx(binmap, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), 1)
    if dilate != (0, 0):
      binmap = cv2.dilate(binmap, cv2.getStructuringElement(cv2.MORPH_RECT, dilate), 1)

    cnts, _ = cv2.findContours(binmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    min_area = max(8, int(0.0003 * nh * nw))
    for c in cnts:
      if cv2.contourArea(c) < min_area:
        continue
      rect = cv2.minAreaRect(c)
      box = cv2.boxPoints(rect).astype(np.float32)

      mask = np.zeros((nh, nw), np.uint8)
      cv2.drawContours(mask, [box.astype(np.int32)], -1, 255, -1)
      score = float(prob_up[mask > 0].mean()) if (mask > 0).any() else 0.0
      if score < box_thresh:
        continue
      boxes.append(box)
    return boxes

  def calc_metrics(self, img_bgr):
    H0, W0 = img_bgr.shape[:2]

    # ---- preprocess (BGR, keep ratio, pad to /32, ImageNet norm) ----
    proc, nh, nw, sh, sw = self._resize_keep_ratio(img_bgr)  # padded image, resized H/W, scales
    x = self._normalize_imagenet(proc)
    x = np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)  # NCHW

    # ---- inference ----
    out = self.sess.run(None, {self.input_name: x})
    prob = out[0]
    while prob.ndim > 2:
      prob = prob[0]  # squeeze to HxW

    # some exports already output [0,1]; if not, apply sigmoid
    pmin, pmax = float(prob.min()), float(prob.max())
    if not (0.0 <= pmin and pmax <= 1.0):
      prob = 1.0 / (1.0 + np.exp(-prob))

    # resize to padded size, then CROP to unpadded (nh, nw)
    prob_full = cv2.resize(prob, (proc.shape[1], proc.shape[0]), interpolation=cv2.INTER_LINEAR)
    prob_up = prob_full[:nh, :nw]  # remove the padding explicitly
    return prob_up, nh, nw, sh, sw

  def infer_boxes(self, prob_up, nh, nw, bin_thresh=0.5, box_thresh=0.6, dilate=(0, 0)) -> list[np.ndarray]:
    """
    Run PP-OCRv3 DB detector on a BGR image and return a list of text quads.
    Returns: [quad(np.float32, shape (4,2)), ...] in original image coords (clockwise).
    """
    boxes = self._find_boxes_from_prob(prob_up, nh, nw, bin_thresh=bin_thresh, box_thresh=box_thresh, dilate=dilate)
    # sort by area (largest first), then return
    if boxes:
      areas = [cv2.contourArea(b.astype(np.float32)) for b in boxes]
      boxes = [b for _, b in sorted(zip(areas, boxes), key=lambda t: t[0], reverse=True)]
    return boxes

# ---------- recognizer (yours, slightly adapted) ----------

class OCRSession:
  def __init__(self,
               onnx_path: str,
               charset_path: str | None = None,
               charset_string: str | None = None,
               fp16_trt: bool = True,
               device_id: int = 0,
               input_height: int = 48,
               input_width: int = 320,
               mean=None,
               std=None,
               blank_idx: int | None = 0,
               cache_dir: str = "trt_cache"):
    self.input_h = input_height
    self.input_w = input_width
    self.mean = mean
    self.std = std

    if charset_path is not None:
      with open(charset_path, "r", encoding="utf-8") as f:
        self.charset = [line.rstrip("\n") for line in f]
    elif charset_string is not None:
      self.charset = list(charset_string)
    else:
      self.charset = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    self.blank_idx = blank_idx if blank_idx is not None else len(self.charset)

    providers = []
    trt_options = {
      "device_id": device_id,
      "trt_max_workspace_size": 1 << 30,
      "trt_fp16_enable": bool(fp16_trt),
      "trt_engine_cache_enable": True,
      "trt_engine_cache_path": cache_dir,
      "trt_timing_cache_enable": True,
      "trt_builder_optimization_level": 5,
    }
    providers.append(("TensorrtExecutionProvider", trt_options))
    providers.append(("CUDAExecutionProvider", {"device_id": device_id}))
    providers.append("CPUExecutionProvider")

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    try:
      self.sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    except Exception:
      print("Failed to launch with TensorRT. Using CUDA...")
      self.sess = ort.InferenceSession(onnx_path, sess_options=so,
        providers=[("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"])
    self.input_name = self.sess.get_inputs()[0].name
    self._warmup()

  def _warmup(self, iters: int = 3) -> None:
    """
    Build TRT plan for the fixed (1,3,48,320) input used by the recognizer.
    """
    x = np.zeros((1, 3, self.input_h, self.input_w), dtype=np.float32)
    for _ in range(iters):
      _ = self.sess.run(None, {self.input_name: x})

  def _preprocess(self, crop_bgr: np.ndarray, center: bool = True, debug: bool = False) -> np.ndarray:
    H, W = self.input_h, self.input_w
    h0, w0 = crop_bgr.shape[:2]
    scale = H / max(1, h0)
    new_w = min(W, int(round(w0 * scale)))
    resized = cv2.resize(crop_bgr, (new_w, H), interpolation=cv2.INTER_LINEAR)
    r, g, b = border_median_color(resized, k=3)
    pad = np.full((H, W, 3), (r, g, b), dtype=np.uint8)
    if center:
      off = (W - new_w) // 2
      pad[:, off:off+new_w] = resized
    else:
      pad[:, :new_w] = resized
    if debug: Image.fromarray(pad).save(r"img_debug\debug_preprocessed.png")
    img = pad.astype(np.float32) / 255.0
    if self.mean is not None and self.std is not None:
      mean = np.array(self.mean, dtype=np.float32).reshape(1, 1, 3)
      std = np.array(self.std, dtype=np.float32).reshape(1, 1, 3)
      img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))[None, ...].astype(np.float32)
    return img

  @staticmethod
  def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    expx = np.exp(x)
    return expx / np.sum(expx, axis=axis, keepdims=True)

  @staticmethod
  def _apply_allowed_mask(logits: np.ndarray, allowed_mask: np.ndarray | None):
    """
    logits: [T, C] or [1, T, C] or [N, T, C]
    allowed_mask: bool array of shape [C] (or [C'] if blank appended per export)
    Sets disallowed class logits to a very negative number.
    """
    if allowed_mask is None:
      return logits
    # Normalize to [T, C]
    if logits.ndim == 3:
      L = logits[0]
      L[:, ~allowed_mask] = -1e9
      logits[0] = L
    elif logits.ndim == 2:
      logits[:, ~allowed_mask] = -1e9
    else:
      # [N, T, C]
      logits[..., ~allowed_mask] = -1e9
    return logits

  def _ctc_greedy_decode(self, logits: np.ndarray) -> tuple[str, float]:
    if logits.ndim == 3:
      logits = logits[0]
    idxs = np.argmax(logits, axis=-1)
    probs = self._softmax(logits, axis=-1)
    maxp = np.max(probs, axis=-1)
    text, conf, prev = [], [], None
    blank = self.blank_idx
    for t, c in enumerate(idxs):
      if c == prev:
        prev = c; continue
      if c != blank and c < len(self.charset):
        text.append(self.charset[c])
        conf.append(float(maxp[t]))
      prev = c
    return ("".join(text).strip(), float(np.mean(conf)) if conf else 0.0)

  def recognize_bgr(self, crop_bgr: np.ndarray, mask: np.ndarray = None, debug: bool = False) -> tuple[str, float]:
    x = self._preprocess(crop_bgr, center=True, debug=debug)
    outputs = self.sess.run(None, {self.input_name: x})
    return self._ctc_greedy_decode(self._apply_allowed_mask(outputs[0], mask))

# ---------- end-to-end: detect-then-recognize ----------

class DetectThenRecognize:
  def __init__(self,
               det_onnx: str,
               rec_onnx: str,
               charset_path: str,
               device_id: int = 0):
    self.det = PPOCRv3Detector(det_onnx, device_id=device_id, use_trt=True)
    self.rec = OCRSession(rec_onnx, charset_path=charset_path,
                          device_id=device_id, input_height=48, input_width=320,
                          mean=None, std=None, blank_idx=0)

  def run(self, rough_crop: np.ndarray | Image.Image, expand: int = 4, mask: np.ndarray = None, min_confidence: float = 0.03, debug: bool = False) -> tuple[str, float]:
    bin_threshs = (0.6, 0.5, 0.4, 0.3)
    box_threshs = (0.7, 0.6, 0.45, 0.3)
    dilates = ((0, 0), (0, 0), (3, 1), (4, 2))

    if isinstance(rough_crop, Image.Image):
      rough_crop = cv2.cvtColor(np.array(rough_crop), cv2.COLOR_RGB2BGR)
    prob_up, nh, nw, sh, sw = self.det.calc_metrics(rough_crop)
    for bin_thresh, box_thresh, dilate in zip(bin_threshs, box_threshs, dilates):
      boxes = self.det.infer_boxes(prob_up, nh, nw, bin_thresh=bin_thresh, box_thresh=box_thresh, dilate=dilate)
      H, W = rough_crop.shape[:2]
      scored = []
      if debug:
        print(f"Image dimensions: {W=}, {H=}")
        print("Boxes: ")
      if boxes:
        for q in boxes:
          x0, y0, x1, y1 = quad_to_aabb(q, W, H, expand=expand)  # small expansion
          x0, y0, x1, y1 = refine_crop_with_prob_segments(x0, y0, x1, y1, prob_up, sh, sw, margin_px=8, gap_px_prob=6, smooth_w=5, min_width_px=12)
          crop = rough_crop[y0:y1, x0:x1, :]
          Image.fromarray(crop).save(r"img_debug\debug_trimmed.png")
          scored.append(result := (*self.rec.recognize_bgr(crop, mask=mask, debug=debug), (x0, y0, x1, y1)))
          if debug: print(f"Box={f'({x0}, {y0}, {x1}, {y1})':<17}  Confidence={result[1]:.4f}  Text={result[0]}")
      else:
        scored.append((*self.rec.recognize_bgr(rough_crop), (0, 0, *rough_crop.shape[:2])))
      best = max(scored, key=lambda t: t[1])
      if best[1] >= min_confidence:
        break
    if debug:
      x0, y0, x1, y1 = best[2]
      Image.fromarray(rough_crop[y0:y1, x0:x1, :]).save(r"img_debug\debug_boxed.png")
    return best[:2]

pipeline = DetectThenRecognize(det_onnx=r"resources\en_PP-OCRv3_det_infer.onnx", rec_onnx=r"resources\en_PP-OCRv3_rec_infer.onnx", charset_path=charset_path)

# ---------- CLI ----------

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("fn", help="roughly cropped image (PNG/JPG)")
  parser.add_argument("--expand", type=int, default=4)
  args = parser.parse_args()

  img_bgr = cv2.imread(args.fn, cv2.IMREAD_COLOR)
  if img_bgr is None:
    print(f"File not found at {args.fn}")
    raise SystemExit(1)
  text, conf = pipeline.run(img_bgr, expand=args.expand, debug=True)
  print(f"Text={text!r}, Confidence={conf:.4f}")
