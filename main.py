import io
import gc
from collections import deque

import numpy as np
from PIL import Image, ImageFilter
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse

app = FastAPI(title="Hybrid Matting API", version="1.4.0")

# Render Free safe defaults
MAX_SIDE_DEFAULT = 2048
BG_MAP_SIDE_DEFAULT = 512


# =========================================================
# Small utils
# =========================================================

def resize_max(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / float(max(w, h))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return img.resize((nw, nh), Image.LANCZOS)

def pil_to_f16(img: Image.Image) -> np.ndarray:
    return np.array(img, dtype=np.uint8).astype(np.float16)

def alpha_to_u8(alpha01: np.ndarray) -> np.ndarray:
    return (np.clip(alpha01, 0.0, 1.0) * 255.0).round().astype(np.uint8)

def u8_to_alpha01(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float16) / np.float16(255.0)

def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()

def max_abs3(a3: np.ndarray) -> np.ndarray:
    # a3: [H,W,3]
    return np.max(np.abs(a3), axis=2)


# =========================================================
# Background-likeness helpers (gating)
# =========================================================

def is_near_white_rgb(px, tol: int) -> bool:
    r, g, b = px
    return abs(r - 255) <= tol and abs(g - 255) <= tol and abs(b - 255) <= tol

def is_near_black_rgb(px, tol: int) -> bool:
    r, g, b = px
    return (r <= tol) and (g <= tol) and (b <= tol)


# =========================================================
# Two-stage floodfill mask from border (memory-safe, 1D)
# strict seeds + relaxed expansion
# =========================================================

def floodfill_border_two_stage_1d(img_key: Image.Image, tol_strict: int, tol_relaxed: int) -> bytearray:
    """
    Returns mask_1d where 1 means "background connected to border" (on the WHITE image).
    """
    w, h = img_key.size
    pix = img_key.load()

    mask = bytearray(w * h)  # 0/1
    q = deque()

    def idx(x, y):
        return y * w + x

    def try_seed_strict(x, y):
        i = idx(x, y)
        if mask[i] == 0 and is_near_white_rgb(pix[x, y], tol_strict):
            mask[i] = 1
            q.append((x, y))

    # strict seeds on border
    for x in range(w):
        try_seed_strict(x, 0)
        try_seed_strict(x, h - 1)
    for y in range(h):
        try_seed_strict(0, y)
        try_seed_strict(w - 1, y)

    # relaxed expansion but only from strict border-connected seeds
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
# Low-res background map for difference matting (gradient-safe, RAM-safe)
# =========================================================

def build_bg_map_lowres(img: Image.Image, blur_radius: float, bg_map_side: int) -> Image.Image:
    if blur_radius <= 0:
        return img
    w, h = img.size
    small = resize_max(img, max_side=bg_map_side)
    small_blur = small.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return small_blur.resize((w, h), Image.BILINEAR)


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
  <head>
    <meta charset="utf-8">
    <title>Hybrid Matting Test</title>
  </head>
  <body style="font-family: Arial; max-width: 980px; margin: 40px auto; line-height:1.45;">
    <h2>Hybrid Matting Test (stable)</h2>
    <p>Upload <b>white</b> and <b>black</b> versions of the same artwork (pixel-aligned).</p>

    <form action="/matte/hybrid?max_side=2048&tol_strict=28&tol_relaxed=60&black_tol=28&bg_blur=12&bg_map_side=512&noise_cut=0.015&denom_min=16&fg_margin=20&debug=0"
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
      Tip: If you ever get an empty PNG, set <b>debug=1</b> to see mask/alpha stats.
    </p>
  </body>
</html>
"""


@app.post("/matte/hybrid")
async def hybrid_matte(
    white_file: UploadFile = File(..., description="Artwork on (near) white background"),
    black_file: UploadFile = File(..., description="Artwork on (near) black background"),

    max_side: int = Query(MAX_SIDE_DEFAULT, ge=512, le=6000),

    # Floodfill two-stage (on WHITE key image)
    tol_strict: int = Query(28, ge=0, le=255),
    tol_relaxed: int = Query(60, ge=0, le=255),

    # Gating using black image bg-likeness too
    black_tol: int = Query(28, ge=0, le=255),

    # Difference matting (inner edge refinement)
    bg_blur: float = Query(12.0, ge=0.0, le=64.0),
    bg_map_side: int = Query(BG_MAP_SIDE_DEFAULT, ge=128, le=1024),
    denom_min: float = Query(16.0, ge=1.0, le=80.0),

    # cleanup
    noise_cut: float = Query(0.015, ge=0.0, le=0.2),

    # anti-wear protection
    fg_margin: float = Query(20.0, ge=0.0, le=120.0),

    # debug
    debug: int = Query(0, ge=0, le=1),
):
    try:
        img_w = Image.open(io.BytesIO(await white_file.read())).convert("RGB")
        img_b = Image.open(io.BytesIO(await black_file.read())).convert("RGB")

        img_w = resize_max(img_w, max_side=max_side)
        img_b = resize_max(img_b, max_side=max_side)

        if img_w.size != img_b.size:
            return JSONResponse(status_code=400, content={"error": "Images must match size (pixel-aligned)."})

        w, h = img_w.size

        # -------------------------------------------------
        # 1) Floodfill mask on a KEY image (reduces JPG noise)
        # -------------------------------------------------
        img_key = img_w.filter(ImageFilter.BoxBlur(0.6))
        if tol_strict >= tol_relaxed:
            tol_strict = max(5, tol_relaxed - 20)

        flood_mask_1d = floodfill_border_two_stage_1d(img_key, tol_strict=tol_strict, tol_relaxed=tol_relaxed)
        flood2d = (np.frombuffer(flood_mask_1d, dtype=np.uint8).reshape(h, w) == 1)

        # -------------------------------------------------
        # 2) Gate floodfill: only accept as BG if it is BG-like
        #    BG-like = near-white (on WHITE key) OR near-black (on BLACK)
        #    This prevents the floodfill from nuking the design.
        # -------------------------------------------------
        # Build bg-like masks cheaply via numpy
        W_u8 = np.array(img_key, dtype=np.uint8)      # key image for whiteness
        B_u8 = np.array(img_b, dtype=np.uint8)        # black image for blackness

        white_like = (np.abs(W_u8.astype(np.int16) - 255).max(axis=2) <= tol_relaxed)
        black_like = (B_u8.max(axis=2) <= black_tol)

        bg2d = flood2d & (white_like | black_like)

        # -------------------------------------------------
        # 3) Difference alpha (conservative)
        # -------------------------------------------------
        W = W_u8.astype(np.float16)
        B = B_u8.astype(np.float16)

        BgW = pil_to_f16(build_bg_map_lowres(img_w, blur_radius=bg_blur, bg_map_side=bg_map_side))
        BgB = pil_to_f16(build_bg_map_lowres(img_b, blur_radius=bg_blur, bg_map_side=bg_map_side))

        # distance-based alpha (more stable for texture)
        diff32 = (W.astype(np.float32) - B.astype(np.float32))
        pix_dist = np.sqrt((diff32 * diff32).sum(axis=2))

        bgdiff32 = (BgW.astype(np.float32) - BgB.astype(np.float32))
        bg_dist = np.sqrt((bgdiff32 * bgdiff32).sum(axis=2))
        bg_dist = np.maximum(bg_dist, float(denom_min))

        alpha = 1.0 - (pix_dist / bg_dist)
        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float16)

        # cleanup temps
        del diff32, pix_dist, bgdiff32, bg_dist
        gc.collect()

        # noise cut (light)
        if noise_cut > 0:
            alpha = np.where(alpha < np.float16(noise_cut), np.float16(0.0), alpha).astype(np.float16)

        # -------------------------------------------------
        # 4) Anti-wear: protect confident foreground pixels
        #    If pixel differs from estimated bg enough -> alpha=1
        # -------------------------------------------------
        if fg_margin > 0:
            dev_w = max_abs3(W - BgW)  # [H,W]
            dev_b = max_abs3(B - BgB)
            fg_conf = (dev_w > np.float16(fg_margin)) | (dev_b > np.float16(fg_margin))
            alpha = np.where((~bg2d) & fg_conf, np.float16(1.0), alpha).astype(np.float16)
            del dev_w, dev_b, fg_conf
            gc.collect()

        # -------------------------------------------------
        # 5) Hybrid merge: enforce outer background as 0 alpha
        # -------------------------------------------------
        alpha_u8 = alpha_to_u8(alpha)
        alpha_u8[bg2d] = 0
        alpha = u8_to_alpha01(alpha_u8)

        # fail-safe: if almost everything is zero, return debug (or fallback)
        nonzero_ratio = float(np.count_nonzero(alpha_u8)) / float(alpha_u8.size)
        bg_ratio = float(np.count_nonzero(bg2d)) / float(bg2d.size)

        if nonzero_ratio < 0.01:
            # fallback: do NOT apply bg2d, only flood2d (stricter) with gating
            # (This avoids "empty PNG" catastrophes.)
            alpha_u8_fallback = np.full((h, w), 255, dtype=np.uint8)
            alpha_u8_fallback[bg2d] = 0
            nonzero_ratio_fb = float(np.count_nonzero(alpha_u8_fallback)) / float(alpha_u8_fallback.size)

            if debug == 1:
                return JSONResponse(status_code=200, content={
                    "warning": "alpha nearly empty; fallback applied",
                    "size": [w, h],
                    "bg_ratio": bg_ratio,
                    "alpha_nonzero_ratio": nonzero_ratio,
                    "fallback_alpha_nonzero_ratio": nonzero_ratio_fb,
                    "tips": [
                        "Lower tol_relaxed (e.g. 55) or black_tol (e.g. 20) if bg_ratio is too high.",
                        "Lower noise_cut (e.g. 0.01) if interiors are being eaten.",
                        "Increase fg_margin (e.g. 24) if design looks worn."
                    ]
                })

            # apply fallback alpha
            alpha_u8 = alpha_u8_fallback
            alpha = u8_to_alpha01(alpha_u8)

        # -------------------------------------------------
        # 6) Color recovery (background-aware)
        # -------------------------------------------------
        a_safe = np.maximum(alpha, np.float16(1e-3))[..., None]
        out_rgb = (B - (np.float16(1.0) - alpha)[..., None] * BgB) / a_safe
        out_rgb = np.clip(out_rgb, 0.0, 255.0)
        out_rgb = np.where(alpha[..., None] <= np.float16(noise_cut), 0.0, out_rgb).astype(np.uint8)

        out = Image.fromarray(out_rgb, mode="RGB").convert("RGBA")
        out.putalpha(Image.fromarray(alpha_u8, mode="L"))

        # free big arrays early
        del W_u8, B_u8, W, B, BgW, BgB, out_rgb, alpha, alpha_u8, flood2d, bg2d, white_like, black_like, flood_mask_1d
        gc.collect()

        return StreamingResponse(io.BytesIO(png_bytes(out)), media_type="image/png")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Processing failed: {e}"})

