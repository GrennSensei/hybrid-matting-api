import io
from collections import deque

import numpy as np
from PIL import Image, ImageFilter
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse

app = FastAPI(title="Hybrid Matting API (Floodfill + Hole Diff RGB Fix)", version="2.3.1")

MAX_SIDE_DEFAULT = 2048


# =========================================================
# Utils
# =========================================================

def resize_max(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / float(max(w, h))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return img.resize((nw, nh), Image.LANCZOS)

def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()

def is_near_white_rgb(px, tol: int) -> bool:
    r, g, b = px
    return abs(r - 255) <= tol and abs(g - 255) <= tol and abs(b - 255) <= tol

def luma_u8(px) -> float:
    r, g, b = px
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


# =========================================================
# Two-stage floodfill from border (memory-safe 1D)
# =========================================================

def floodfill_border_bg_mask_two_stage_1d(img_for_keying: Image.Image, tol_strict: int, tol_relaxed: int) -> bytearray:
    w, h = img_for_keying.size
    pix = img_for_keying.load()

    mask = bytearray(w * h)  # 0/1
    q = deque()

    def idx(x, y): return y * w + x

    def try_seed_strict(x, y):
        i = idx(x, y)
        if mask[i] == 0 and is_near_white_rgb(pix[x, y], tol_strict):
            mask[i] = 1
            q.append((x, y))

    for x in range(w):
        try_seed_strict(x, 0)
        try_seed_strict(x, h - 1)
    for y in range(h):
        try_seed_strict(0, y)
        try_seed_strict(w - 1, y)

    while q:
        x, y = q.popleft()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < w and 0 <= ny < h:
                i = idx(nx, ny)
                if mask[i] == 0 and is_near_white_rgb(pix[nx, ny], tol_relaxed):
                    mask[i] = 1
                    q.append((nx, ny))

    return mask


# =========================================================
# Find enclosed transparent holes (inside FG) using 1D BFS
# =========================================================

def find_enclosed_holes(alpha_bg_1d: bytearray, w: int, h: int, min_area: int):
    def idx(x, y): return y * w + x

    visited = bytearray(w * h)
    q = deque()

    def enqueue_if_bg(x, y):
        i = idx(x, y)
        if visited[i] == 0 and alpha_bg_1d[i] == 1:
            visited[i] = 1
            q.append((x, y))

    for x in range(w):
        enqueue_if_bg(x, 0)
        enqueue_if_bg(x, h - 1)
    for y in range(h):
        enqueue_if_bg(0, y)
        enqueue_if_bg(w - 1, y)

    while q:
        x, y = q.popleft()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < w and 0 <= ny < h:
                i = idx(nx, ny)
                if visited[i] == 0 and alpha_bg_1d[i] == 1:
                    visited[i] = 1
                    q.append((nx, ny))

    holes = []
    hole_vis = bytearray(w * h)

    for y0 in range(h):
        for x0 in range(w):
            i0 = idx(x0, y0)
            if alpha_bg_1d[i0] != 1:
                continue
            if visited[i0] == 1:
                continue
            if hole_vis[i0] == 1:
                continue

            comp = []
            qq = deque([(x0, y0)])
            hole_vis[i0] = 1

            while qq:
                x, y = qq.popleft()
                comp.append(idx(x, y))
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if 0 <= nx < w and 0 <= ny < h:
                        ii = idx(nx, ny)
                        if alpha_bg_1d[ii] == 1 and visited[ii] == 0 and hole_vis[ii] == 0:
                            hole_vis[ii] = 1
                            qq.append((nx, ny))

            if len(comp) >= min_area:
                holes.append(comp)

    return holes


# =========================================================
# Difference alpha + BG estimate
# =========================================================

def estimate_bg_rgb_from_border_mask(img_rgb_u8: np.ndarray, border_bg_1d: bytearray) -> np.ndarray:
    h, w, _ = img_rgb_u8.shape
    m = np.frombuffer(border_bg_1d, dtype=np.uint8).reshape(h, w) == 1
    if not np.any(m):
        corners = np.array([img_rgb_u8[0, 0], img_rgb_u8[0, w - 1], img_rgb_u8[h - 1, 0], img_rgb_u8[h - 1, w - 1]], dtype=np.float32)
        return np.median(corners, axis=0)
    pix = img_rgb_u8[m].astype(np.float32)
    return np.median(pix, axis=0)

def compute_alpha_dist(W_u8: np.ndarray, B_u8: np.ndarray, bgW: np.ndarray, bgB: np.ndarray) -> np.ndarray:
    W = W_u8.astype(np.float32)
    B = B_u8.astype(np.float32)
    diff = W - B
    pix_dist = np.sqrt((diff * diff).sum(axis=2))  # [H,W]

    bg_diff = bgW.astype(np.float32) - bgB.astype(np.float32)
    bg_dist = float(np.sqrt((bg_diff * bg_diff).sum()))
    bg_dist = max(bg_dist, 1.0)

    alpha = 1.0 - (pix_dist / bg_dist)
    return np.clip(alpha, 0.0, 1.0)


# =========================================================
# Edge-safe fringe killer (keeps contours)
# =========================================================

def fringe_kill_near_white_edge_safe(alpha_bg_1d: bytearray, img_w_rgb: Image.Image, tol: int, passes: int, protect_dark_luma: float) -> int:
    if passes <= 0:
        return 0

    w, h = img_w_rgb.size
    pix = img_w_rgb.load()

    def idx(x, y): return y * w + x

    flipped_total = 0
    for _ in range(passes):
        to_flip = []
        for y in range(h):
            for x in range(w):
                i = idx(x, y)
                if alpha_bg_1d[i] == 1:
                    continue

                px = pix[x, y]
                if luma_u8(px) < protect_dark_luma:
                    continue
                if not is_near_white_rgb(px, tol):
                    continue

                touch_bg = False
                if x > 0 and alpha_bg_1d[idx(x - 1, y)] == 1: touch_bg = True
                elif x < w - 1 and alpha_bg_1d[idx(x + 1, y)] == 1: touch_bg = True
                elif y > 0 and alpha_bg_1d[idx(x, y - 1)] == 1: touch_bg = True
                elif y < h - 1 and alpha_bg_1d[idx(x, y + 1)] == 1: touch_bg = True

                if touch_bg:
                    to_flip.append(i)

        if not to_flip:
            break

        for i in to_flip:
            alpha_bg_1d[i] = 1
        flipped_total += len(to_flip)

    return flipped_total


# =========================================================
# Routes
# =========================================================

@app.get("/health")
def health():
    return {"status": "ok", "endpoint": "/matte/hybrid"}

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html>
<html>
  <head><meta charset="utf-8"><title>Hybrid Matting Test</title></head>
  <body style="font-family: Arial; max-width: 980px; margin: 40px auto; line-height:1.45;">
    <h2>Hybrid Matting Test (Final defaults)</h2>
    <p>
      Külső háttér: two-stage floodfill (BoxBlur 0.6) + edge-safe fringe fix.<br/>
      Hole-ok: difference alapján nem csak alpha-t, hanem RGB-t is javítunk (csak hole pixelekben).
    </p>

    <form action="/matte/hybrid?max_side=2048&tol_strict=35&tol_relaxed=60&fringe_tol=55&fringe_passes=1&protect_dark_luma=130&hole_fill=1&min_hole_area=100&hole_alpha_thresh=0.14&hole_alpha_min=0.75&debug=0"
          method="post" enctype="multipart/form-data"
          style="padding:16px; border:1px solid #ddd; border-radius:10px;">
      <p><b>White image:</b></p>
      <input type="file" name="white_file" accept="image/*" required />

      <p style="margin-top:14px;"><b>Black image:</b></p>
      <input type="file" name="black_file" accept="image/*" required />

      <p style="margin-top:14px;">
        <button type="submit" style="padding:10px 14px; font-weight:bold; cursor:pointer;">
          Generate Transparent PNG
        </button>
      </p>
    </form>

    <p style="color:#777; margin-top:14px;">
      Ha hole-ban még látszik háttér: emeld <b>hole_alpha_min</b>-t (0.75→0.85) vagy csökkentsd <b>hole_alpha_thresh</b>-t (0.14→0.12).<br/>
      Ha rossz helyen is kitölt: emeld <b>hole_alpha_thresh</b>-t (0.14→0.18) vagy emeld <b>min_hole_area</b>-t (100→160).
    </p>
  </body>
</html>
"""


@app.post("/matte/hybrid")
async def hybrid_matte(
    white_file: UploadFile = File(...),
    black_file: UploadFile = File(...),

    max_side: int = Query(MAX_SIDE_DEFAULT, ge=512, le=6000),

    tol_strict: int = Query(35, ge=0, le=255),
    tol_relaxed: int = Query(60, ge=0, le=255),

    fringe_tol: int = Query(55, ge=0, le=255),
    fringe_passes: int = Query(1, ge=0, le=4),
    protect_dark_luma: float = Query(130.0, ge=0.0, le=255.0),

    hole_fill: int = Query(1, ge=0, le=1),

    # ✅ Updated defaults (as requested)
    min_hole_area: int = Query(100, ge=1, le=500000),
    hole_alpha_thresh: float = Query(0.14, ge=0.0, le=1.0),
    hole_alpha_min: float = Query(0.75, ge=0.05, le=1.0),

    debug: int = Query(0, ge=0, le=1),
):
    try:
        img_w = Image.open(io.BytesIO(await white_file.read())).convert("RGB")
        img_b = Image.open(io.BytesIO(await black_file.read())).convert("RGB")

        img_w = resize_max(img_w, max_side=max_side)
        img_b = resize_max(img_b, max_side=max_side)

        if img_w.size != img_b.size:
            return JSONResponse(status_code=400, content={"error": "A két kép mérete nem egyezik (pixelpontos egyezés kell)."})


        w, h = img_w.size

        # 1) floodfill outer background
        img_key = img_w.filter(ImageFilter.BoxBlur(0.6))
        if tol_strict >= tol_relaxed:
            tol_strict = max(10, tol_relaxed - 30)

        border_bg_1d = floodfill_border_bg_mask_two_stage_1d(img_key, tol_strict=tol_strict, tol_relaxed=tol_relaxed)

        # alpha_bg_1d: 1=bg, 0=fg
        alpha_bg_1d = border_bg_1d[:]

        # 2) prepare arrays
        W_u8 = np.array(img_w, dtype=np.uint8)
        B_u8 = np.array(img_b, dtype=np.uint8)

        # Output RGB starts from WHITE (no global color recovery)
        out_rgb = W_u8.copy()

        # 3) hole fill using difference + RGB recovery ONLY for hole pixels
        hole_components = 0
        hole_pixels_filled = 0
        hole_pixels_rgb_fixed = 0

        if hole_fill == 1:
            holes = find_enclosed_holes(alpha_bg_1d, w, h, min_area=min_hole_area)
            hole_components = len(holes)

            if hole_components > 0:
                bgW = estimate_bg_rgb_from_border_mask(W_u8, border_bg_1d)
                bgB = estimate_bg_rgb_from_border_mask(B_u8, border_bg_1d)

                alpha_diff = compute_alpha_dist(W_u8, B_u8, bgW=bgW, bgB=bgB)  # [H,W] float32

                bgB_f = bgB.astype(np.float32)

                for comp in holes:
                    for ii in comp:
                        x = ii % w
                        y = ii // w

                        a = float(alpha_diff[y, x])
                        if a < hole_alpha_thresh:
                            continue

                        # Mark as foreground (opaque)
                        alpha_bg_1d[ii] = 0
                        hole_pixels_filled += 1

                        # RGB recovery ONLY here
                        a_use = max(a, hole_alpha_min)

                        Bpx = B_u8[y, x].astype(np.float32)
                        C = (Bpx - (1.0 - a_use) * bgB_f) / a_use
                        C = np.clip(C, 0.0, 255.0).astype(np.uint8)

                        out_rgb[y, x] = C
                        hole_pixels_rgb_fixed += 1

        # 4) edge-safe fringe killer
        flipped_fringe = fringe_kill_near_white_edge_safe(
            alpha_bg_1d, img_w, tol=fringe_tol, passes=fringe_passes, protect_dark_luma=protect_dark_luma
        )

        # 5) build RGBA
        out = Image.fromarray(out_rgb, mode="RGB").convert("RGBA")
        p = out.load()

        i = 0
        for y in range(h):
            for x in range(w):
                r, g, b, _ = p[x, y]
                a = 0 if alpha_bg_1d[i] == 1 else 255
                p[x, y] = (r, g, b, a)
                i += 1

        if debug == 1:
            bg_ratio = float(sum(alpha_bg_1d)) / float(len(alpha_bg_1d))
            return JSONResponse(status_code=200, content={
                "size": [w, h],
                "bg_ratio": bg_ratio,
                "hole_components": hole_components,
                "hole_pixels_filled": hole_pixels_filled,
                "hole_pixels_rgb_fixed": hole_pixels_rgb_fixed,
                "fringe_pixels_removed": flipped_fringe,
                "params": {
                    "tol_strict": tol_strict,
                    "tol_relaxed": tol_relaxed,
                    "fringe_tol": fringe_tol,
                    "protect_dark_luma": protect_dark_luma,
                    "hole_alpha_thresh": hole_alpha_thresh,
                    "hole_alpha_min": hole_alpha_min,
                    "min_hole_area": min_hole_area
                }
            })

        return StreamingResponse(io.BytesIO(png_bytes(out)), media_type="image/png")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Processing failed: {e}"})
