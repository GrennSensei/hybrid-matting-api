import io
from collections import deque

import numpy as np
from PIL import Image, ImageFilter
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse

app = FastAPI(title="Hybrid Matting API (Floodfill + Hole Diff)", version="2.0.0")

# Render Free safe defaults
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

def is_near_white(px, tol: int) -> bool:
    r, g, b = px
    return abs(r - 255) <= tol and abs(g - 255) <= tol and abs(b - 255) <= tol


# =========================================================
# Two-stage floodfill from border (memory-safe 1D)
# strict seeds + relaxed expansion
# =========================================================

def floodfill_border_bg_mask_two_stage_1d(
    img_for_keying: Image.Image,
    tol_strict: int,
    tol_relaxed: int,
) -> bytearray:
    w, h = img_for_keying.size
    pix = img_for_keying.load()

    mask = bytearray(w * h)  # 0/1
    q = deque()

    def idx(x, y): return y * w + x

    def try_seed_strict(x, y):
        i = idx(x, y)
        if mask[i] == 0 and is_near_white(pix[x, y], tol_strict):
            mask[i] = 1
            q.append((x, y))

    # strict border seeds
    for x in range(w):
        try_seed_strict(x, 0)
        try_seed_strict(x, h - 1)
    for y in range(h):
        try_seed_strict(0, y)
        try_seed_strict(w - 1, y)

    # relaxed expansion from strict seeds
    while q:
        x, y = q.popleft()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < w and 0 <= ny < h:
                i = idx(nx, ny)
                if mask[i] == 0 and is_near_white(pix[nx, ny], tol_relaxed):
                    mask[i] = 1
                    q.append((nx, ny))

    return mask


# =========================================================
# Find enclosed transparent holes (inside FG) using 1D BFS
# Holes = transparent pixels that are NOT border-connected background
# We return components (list of indices) for holes only.
# =========================================================

def find_enclosed_holes(alpha_mask_1d: bytearray, w: int, h: int, min_area: int):
    """
    alpha_mask_1d: 1 where background (transparent), 0 where foreground (opaque)
    We consider holes as background pixels that are NOT connected to border.
    Returns list of hole components, each as list of 1D indices.
    """
    def idx(x, y): return y * w + x

    # visited for background pixels
    visited = bytearray(w * h)

    # Step 1: mark border-connected background pixels
    q = deque()

    def enqueue_if_bg(x, y):
        i = idx(x, y)
        if visited[i] == 0 and alpha_mask_1d[i] == 1:
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
                if visited[i] == 0 and alpha_mask_1d[i] == 1:
                    visited[i] = 1
                    q.append((nx, ny))

    # Step 2: any remaining background pixels (alpha_mask_1d==1 and visited==0) are enclosed holes
    holes = []
    hole_vis = bytearray(w * h)

    for y0 in range(h):
        for x0 in range(w):
            i0 = idx(x0, y0)
            if alpha_mask_1d[i0] != 1:
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
                        if alpha_mask_1d[ii] == 1 and visited[ii] == 0 and hole_vis[ii] == 0:
                            hole_vis[ii] = 1
                            qq.append((nx, ny))

            if len(comp) >= min_area:
                holes.append(comp)

    return holes


# =========================================================
# Difference alpha (simple + robust enough for HOLES only)
# We estimate background colors from border-connected background pixels.
# =========================================================

def estimate_bg_rgb_from_mask(img_rgb_u8: np.ndarray, bg_mask_1d: bytearray) -> np.ndarray:
    """
    img_rgb_u8: [H,W,3] uint8
    bg_mask_1d: 1 where border-connected background
    Returns bg_rgb float32 shape [3] as median of bg pixels (or fallback).
    """
    h, w, _ = img_rgb_u8.shape
    m = np.frombuffer(bg_mask_1d, dtype=np.uint8).reshape(h, w) == 1
    if not np.any(m):
        # fallback: just use corners median-ish
        corners = np.array([
            img_rgb_u8[0, 0], img_rgb_u8[0, w - 1],
            img_rgb_u8[h - 1, 0], img_rgb_u8[h - 1, w - 1]
        ], dtype=np.float32)
        return np.median(corners, axis=0)
    pix = img_rgb_u8[m].astype(np.float32)
    return np.median(pix, axis=0)

def compute_alpha_dist(W_u8: np.ndarray, B_u8: np.ndarray, bgW: np.ndarray, bgB: np.ndarray) -> np.ndarray:
    """
    Returns alpha in [0,1] float32 using distance formula:
    alpha = 1 - ||W-B|| / ||bgW-bgB||
    """
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
    <h2>Hybrid Matting Test (Floodfill outer + Diff holes)</h2>
    <p>
      Feltöltöd a <b>fehér</b> és a <b>fekete</b> hátteres képet (pixelpontos egyezés).
      <br/>
      Külső háttér: bevált floodfill. Belső lyukak: difference matting csak ott, ahol kell.
    </p>

    <form action="/matte/hybrid?max_side=2048&tol_strict=28&tol_relaxed=60&hole_fill=1&min_hole_area=120&hole_alpha_thresh=0.18&debug=0"
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
      Ha belül “lyukak” maradnak: emeld <b>hole_alpha_thresh</b>-t (0.18→0.22) vagy csökkentsd <b>min_hole_area</b>-t (120→60).
      <br/>
      Ha a minta belsejét is kitölti feleslegesen: emeld <b>min_hole_area</b>-t (120→250).
    </p>
  </body>
</html>
"""


@app.post("/matte/hybrid")
async def hybrid_matte(
    white_file: UploadFile = File(...),
    black_file: UploadFile = File(...),

    max_side: int = Query(MAX_SIDE_DEFAULT, ge=512, le=6000),

    # Floodfill settings (outer bg)
    tol_strict: int = Query(28, ge=0, le=255),
    tol_relaxed: int = Query(60, ge=0, le=255),

    # Hole fill via difference
    hole_fill: int = Query(1, ge=0, le=1),
    min_hole_area: int = Query(120, ge=1, le=500000),
    hole_alpha_thresh: float = Query(0.18, ge=0.0, le=1.0),

    # Debug
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

        # --- Floodfill outer background (BEVÁLT IRÁNY) ---
        img_key = img_w.filter(ImageFilter.BoxBlur(0.6))
        if tol_strict >= tol_relaxed:
            tol_strict = max(5, tol_relaxed - 20)

        border_bg_1d = floodfill_border_bg_mask_two_stage_1d(
            img_for_keying=img_key,
            tol_strict=tol_strict,
            tol_relaxed=tol_relaxed,
        )

        # alpha_mask_1d: 1 for bg (transparent), 0 for fg (opaque)
        alpha_mask_1d = border_bg_1d[:]  # copy

        # --- Optional: fill enclosed holes using difference alpha only on holes ---
        filled_holes = 0
        hole_components = 0

        W_u8 = np.array(img_w, dtype=np.uint8)
        B_u8 = np.array(img_b, dtype=np.uint8)

        if hole_fill == 1:
            holes = find_enclosed_holes(alpha_mask_1d, w, h, min_area=min_hole_area)
            hole_components = len(holes)

            if hole_components > 0:
                # estimate background colors from BORDER-connected background pixels
                bgW = estimate_bg_rgb_from_mask(W_u8, border_bg_1d)
                bgB = estimate_bg_rgb_from_mask(B_u8, border_bg_1d)

                # compute alpha_diff for whole image (cheap enough at 2K)
                alpha_diff = compute_alpha_dist(W_u8, B_u8, bgW=bgW, bgB=bgB)  # [H,W] float32

                # For each hole component, decide fill pixel-by-pixel using alpha_diff threshold
                for comp in holes:
                    for ii in comp:
                        x = ii % w
                        y = ii // w
                        if alpha_diff[y, x] >= hole_alpha_thresh:
                            # fill this hole pixel => make it opaque
                            alpha_mask_1d[ii] = 0
                            filled_holes += 1

        # --- Build output RGBA using WHITE image colors (NO color recovery = NO distortion) ---
        out = img_w.convert("RGBA")
        p = out.load()

        i = 0
        for y in range(h):
            for x in range(w):
                r, g, b, _ = p[x, y]
                a = 0 if alpha_mask_1d[i] == 1 else 255
                p[x, y] = (r, g, b, a)
                i += 1

        if debug == 1:
            bg_ratio = float(sum(alpha_mask_1d)) / float(len(alpha_mask_1d))
            return JSONResponse(status_code=200, content={
                "size": [w, h],
                "bg_ratio": bg_ratio,
                "hole_components": hole_components,
                "hole_pixels_filled": filled_holes,
                "params": {
                    "max_side": max_side,
                    "tol_strict": tol_strict,
                    "tol_relaxed": tol_relaxed,
                    "hole_fill": hole_fill,
                    "min_hole_area": min_hole_area,
                    "hole_alpha_thresh": hole_alpha_thresh,
                },
                "tips": {
                    "if_outer_bg_is_perfect_keep_it": "Külső háttérnél csak a tol_relaxed-et finomítsd (55–75).",
                    "if_holes_remain": "Csökkentsd min_hole_area-t (120→60) vagy csökkentsd hole_alpha_thresh-et (0.18→0.14).",
                    "if_false_fills_inside_design": "Emeld min_hole_area-t (120→250) vagy emeld hole_alpha_thresh-et (0.18→0.22)."
                }
            })

        return StreamingResponse(io.BytesIO(png_bytes(out)), media_type="image/png")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Processing failed: {e}"})
