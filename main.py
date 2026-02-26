import io
from collections import deque

import numpy as np
from PIL import Image, ImageFilter
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse

app = FastAPI(title="Hybrid Matting API (Floodfill + Hole Diff + Edge-safe Fringe Fix)", version="2.2.0")

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
    # Perceptual-ish luminance
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
# Difference alpha (used ONLY for holes decision)
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
# Edge-safe fringe killer
# - only near-white pixels adjacent to bg
# - BUT protect dark contour pixels by luma threshold
# =========================================================

def fringe_kill_near_white_edge_safe(
    alpha_bg_1d: bytearray,
    img_w_rgb: Image.Image,
    tol: int,
    passes: int,
    protect_dark_luma: float,
) -> int:
    """
    alpha_bg_1d: 1 = bg (transparent), 0 = fg (opaque)
    Flip FG->BG only if:
      - near-white (tol)
      - touches BG
      - AND NOT "dark" (luma >= protect_dark_luma)
    """
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
                    continue  # already bg

                px = pix[x, y]

                # PROTECT contours: if dark enough, never delete it
                if luma_u8(px) < protect_dark_luma:
                    continue

                # candidate only if near-white
                if not is_near_white_rgb(px, tol):
                    continue

                # must touch background
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
    <h2>Hybrid Matting Test (Sharp contours)</h2>
    <p>
      Külső háttér: two-stage floodfill (BoxBlur 0.6).<br/>
      Lyukak: difference csak a belső hole pixelekre.<br/>
      Kontúr-élesség: edge-safe fringe killer (nem harap bele a sötét kontúrba).
    </p>

    <form action="/matte/hybrid?max_side=2048&tol_strict=35&tol_relaxed=60&erode_px=0&fringe_tol=55&fringe_passes=1&protect_dark_luma=130&hole_fill=1&min_hole_area=120&hole_alpha_thresh=0.18&debug=0"
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
      Ha még harap: emeld <b>protect_dark_luma</b>-t (130→150), vagy csökkentsd <b>fringe_tol</b>-t (55→45).<br/>
      Ha visszajön a fehér csík: csökkentsd <b>protect_dark_luma</b>-t (130→110) vagy emeld <b>fringe_tol</b>-t (55→70).
    </p>
  </body>
</html>
"""


@app.post("/matte/hybrid")
async def hybrid_matte(
    white_file: UploadFile = File(...),
    black_file: UploadFile = File(...),

    max_side: int = Query(MAX_SIDE_DEFAULT, ge=512, le=6000),

    # outer bg remover (proven setup)
    tol_strict: int = Query(35, ge=0, le=255),
    tol_relaxed: int = Query(60, ge=0, le=255),
    erode_px: int = Query(0, ge=0, le=6),  # DEFAULT 0 to avoid contour bite

    # fringe killer
    fringe_tol: int = Query(55, ge=0, le=255),
    fringe_passes: int = Query(1, ge=0, le=4),
    protect_dark_luma: float = Query(130.0, ge=0.0, le=255.0),

    # holes via difference
    hole_fill: int = Query(1, ge=0, le=1),
    min_hole_area: int = Query(120, ge=1, le=500000),
    hole_alpha_thresh: float = Query(0.18, ge=0.0, le=1.0),

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

        # 1) floodfill on key image (BoxBlur 0.6)
        img_key = img_w.filter(ImageFilter.BoxBlur(0.6))
        if tol_strict >= tol_relaxed:
            tol_strict = max(10, tol_relaxed - 30)

        border_bg_1d = floodfill_border_bg_mask_two_stage_1d(img_key, tol_strict=tol_strict, tol_relaxed=tol_relaxed)
        alpha_bg_1d = border_bg_1d[:]  # 1=bg, 0=fg

        # 2) optional hole fill using difference (ONLY inside enclosed holes)
        filled_hole_px = 0
        hole_components = 0

        W_u8 = np.array(img_w, dtype=np.uint8)
        B_u8 = np.array(img_b, dtype=np.uint8)

        if hole_fill == 1:
            holes = find_enclosed_holes(alpha_bg_1d, w, h, min_area=min_hole_area)
            hole_components = len(holes)

            if hole_components > 0:
                bgW = estimate_bg_rgb_from_border_mask(W_u8, border_bg_1d)
                bgB = estimate_bg_rgb_from_border_mask(B_u8, border_bg_1d)
                alpha_diff = compute_alpha_dist(W_u8, B_u8, bgW=bgW, bgB=bgB)

                for comp in holes:
                    for ii in comp:
                        x = ii % w
                        y = ii // w
                        if alpha_diff[y, x] >= hole_alpha_thresh:
                            alpha_bg_1d[ii] = 0
                            filled_hole_px += 1

        # 3) optional erode (OFF by default to keep contours sharp)
        if erode_px > 0:
            def idx(x, y): return y * w + x

            fg = bytearray(w * h)
            for i in range(w * h):
                fg[i] = 1 if alpha_bg_1d[i] == 0 else 0

            for _ in range(erode_px):
                to_clear = []
                for y in range(h):
                    for x in range(w):
                        i = idx(x, y)
                        if fg[i] == 1:
                            if (x > 0 and fg[idx(x - 1, y)] == 0) or (x < w - 1 and fg[idx(x + 1, y)] == 0) or \
                               (y > 0 and fg[idx(x, y - 1)] == 0) or (y < h - 1 and fg[idx(x, y + 1)] == 0):
                                to_clear.append(i)
                for i in to_clear:
                    fg[i] = 0

            for i in range(w * h):
                alpha_bg_1d[i] = 0 if fg[i] == 1 else 1

        # 4) edge-safe fringe kill (removes white halo WITHOUT biting dark contour)
        flipped_fringe = fringe_kill_near_white_edge_safe(
            alpha_bg_1d, img_w, tol=fringe_tol, passes=fringe_passes, protect_dark_luma=protect_dark_luma
        )

        # 5) build RGBA using WHITE colors (no color recovery => no distortion)
        out = img_w.convert("RGBA")
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
                "hole_pixels_filled": filled_hole_px,
                "fringe_pixels_removed": flipped_fringe,
                "params": {
                    "tol_strict": tol_strict,
                    "tol_relaxed": tol_relaxed,
                    "erode_px": erode_px,
                    "fringe_tol": fringe_tol,
                    "fringe_passes": fringe_passes,
                    "protect_dark_luma": protect_dark_luma,
                    "hole_alpha_thresh": hole_alpha_thresh,
                    "min_hole_area": min_hole_area
                },
                "tuning": {
                    "if_contour_bitten": "protect_dark_luma 130→150 OR fringe_tol 55→45",
                    "if_white_halo_returns": "protect_dark_luma 130→110 OR fringe_tol 55→70"
                }
            })

        return StreamingResponse(io.BytesIO(png_bytes(out)), media_type="image/png")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Processing failed: {e}"})
