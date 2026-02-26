import io
import gc
from collections import deque

import numpy as np
from PIL import Image, ImageFilter
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse

app = FastAPI(title="Hybrid Matting API", version="1.5.0")

# Render Free safe defaults
MAX_SIDE_DEFAULT = 2048
BG_MAP_SIDE_DEFAULT = 512


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

def alpha_to_u8(alpha01: np.ndarray) -> np.ndarray:
    return (np.clip(alpha01, 0.0, 1.0) * 255.0).round().astype(np.uint8)

def u8_to_alpha01(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float16) / np.float16(255.0)

def max_abs3(a3: np.ndarray) -> np.ndarray:
    return np.max(np.abs(a3), axis=2)

def build_bg_map_lowres(img: Image.Image, blur_radius: float, bg_map_side: int) -> Image.Image:
    if blur_radius <= 0:
        return img
    w, h = img.size
    small = resize_max(img, max_side=bg_map_side)
    small_blur = small.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return small_blur.resize((w, h), Image.BILINEAR)


# =========================================================
# Floodfill (two-stage) on WHITE key image
# =========================================================

def is_near_white_rgb(px, tol: int) -> bool:
    r, g, b = px
    return abs(r - 255) <= tol and abs(g - 255) <= tol and abs(b - 255) <= tol

def is_near_black_rgb(px, tol: int) -> bool:
    r, g, b = px
    return (r <= tol) and (g <= tol) and (b <= tol)

def floodfill_border_two_stage_1d(img_key: Image.Image, tol_strict: int, tol_relaxed: int) -> bytearray:
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

    # strict seeds
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
                if mask[i] == 0 and is_near_white_rgb(pix[nx, ny], tol_relaxed):
                    mask[i] = 1
                    q.append((nx, ny))

    return mask


# =========================================================
# Routes
# =========================================================

@app.get("/health")
def health():
    return {"status": "ok", "endpoint": "/matte/hybrid"}

@app.get("/", response_class=HTMLResponse)
def home(debug: int = Query(0, ge=0, le=1)):
    # IMPORTANT: propagate debug into the form action
    # so opening "/?debug=1" actually triggers debug in POST.
    action = (
        f"/matte/hybrid?"
        f"max_side=2048&tol_strict=28&tol_relaxed=60&black_tol=28&"
        f"bg_blur=12&bg_map_side=512&denom_min=16&noise_cut=0.012&"
        f"fg_margin=22&alpha_hard=0.97&debug={debug}"
    )

    return f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Hybrid Matting Test</title>
  </head>
  <body style="font-family: Arial; max-width: 980px; margin: 40px auto; line-height:1.45;">
    <h2>Hybrid Matting Test</h2>
    <p>
      Tölts fel egy <b>fehér</b> és egy <b>fekete</b> hátteres verziót (pixelpontos egyezés).
    </p>

    <div style="padding:12px; border:1px solid #eee; border-radius:10px; background:#fafafa; margin-bottom:14px;">
      <b>Debug mód:</b> jelenleg <b>{'BE' if debug==1 else 'KI'}</b>.
      <br/>
      <span style="color:#555;">
        Debug módban a “Generate” után <b>nem PNG</b> jön, hanem egy szöveg (statisztika), ami segít hibát keresni.
      </span>
      <br/>
      <span style="color:#555;">
        Debug BE: nyisd így: <code>/?debug=1</code> • Debug KI: <code>/</code>
      </span>
    </div>

    <form action="{action}"
          method="post" enctype="multipart/form-data"
          style="padding:16px; border:1px solid #ddd; border-radius:10px;">
      <p><b>White image:</b></p>
      <input type="file" name="white_file" accept="image/*" required />

      <p style="margin-top:14px;"><b>Black image:</b></p>
      <input type="file" name="black_file" accept="image/*" required />

      <p style="margin-top:14px;">
        <button type="submit" style="padding:10px 14px; font-weight:bold; cursor:pointer;">
          Generate
        </button>
      </p>
    </form>

    <p style="color:#777; margin-top:14px;">
      Ha “kopott” a minta: növeld <b>fg_margin</b>-t (22→28), vagy csökkentsd <b>noise_cut</b>-ot (0.012→0.008).
      <br/>
      Ha sötétedik/feketedik: emeld <b>alpha_hard</b>-ot (0.97→0.985).
    </p>
  </body>
</html>
"""


@app.post("/matte/hybrid")
async def hybrid_matte(
    white_file: UploadFile = File(...),
    black_file: UploadFile = File(...),

    max_side: int = Query(MAX_SIDE_DEFAULT, ge=512, le=6000),

    # Floodfill
    tol_strict: int = Query(28, ge=0, le=255),
    tol_relaxed: int = Query(60, ge=0, le=255),
    black_tol: int = Query(28, ge=0, le=255),

    # Difference
    bg_blur: float = Query(12.0, ge=0.0, le=64.0),
    bg_map_side: int = Query(BG_MAP_SIDE_DEFAULT, ge=128, le=1024),
    denom_min: float = Query(16.0, ge=1.0, le=120.0),

    # Cleanup
    noise_cut: float = Query(0.012, ge=0.0, le=0.2),

    # Anti-wear
    fg_margin: float = Query(22.0, ge=0.0, le=120.0),

    # Anti-dark: if alpha is already "almost opaque", don't unpremultiply; just keep original color
    alpha_hard: float = Query(0.97, ge=0.5, le=1.0),

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

        # 1) Floodfill on key image (reduces JPG noise)
        img_key = img_w.filter(ImageFilter.BoxBlur(0.6))
        if tol_strict >= tol_relaxed:
            tol_strict = max(5, tol_relaxed - 20)

        flood_mask_1d = floodfill_border_two_stage_1d(img_key, tol_strict=tol_strict, tol_relaxed=tol_relaxed)
        flood2d = (np.frombuffer(flood_mask_1d, dtype=np.uint8).reshape(h, w) == 1)

        # 2) Gating bg: only accept flood bg where it also looks like bg in white or black
        W_key_u8 = np.array(img_key, dtype=np.uint8)   # only for whiteness check
        W_u8 = np.array(img_w, dtype=np.uint8)         # ORIGINAL white (for color/alpha logic)
        B_u8 = np.array(img_b, dtype=np.uint8)         # ORIGINAL black

        white_like = (np.abs(W_key_u8.astype(np.int16) - 255).max(axis=2) <= tol_relaxed)
        black_like = (B_u8.max(axis=2) <= black_tol)

        bg2d = flood2d & (white_like | black_like)

        # 3) Background maps
        BgW_u8 = np.array(build_bg_map_lowres(img_w, blur_radius=bg_blur, bg_map_side=bg_map_side), dtype=np.uint8)
        BgB_u8 = np.array(build_bg_map_lowres(img_b, blur_radius=bg_blur, bg_map_side=bg_map_side), dtype=np.uint8)

        W = W_u8.astype(np.float16)
        B = B_u8.astype(np.float16)
        BgW = BgW_u8.astype(np.float16)
        BgB = BgB_u8.astype(np.float16)

        # 4) Difference alpha (distance-based, stable)
        diff32 = (W.astype(np.float32) - B.astype(np.float32))
        pix_dist = np.sqrt((diff32 * diff32).sum(axis=2))

        bgdiff32 = (BgW.astype(np.float32) - BgB.astype(np.float32))
        bg_dist = np.sqrt((bgdiff32 * bgdiff32).sum(axis=2))
        bg_dist = np.maximum(bg_dist, float(denom_min))

        alpha = 1.0 - (pix_dist / bg_dist)
        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float16)

        # 5) Light noise cut
        if noise_cut > 0:
            alpha = np.where(alpha < np.float16(noise_cut), np.float16(0.0), alpha).astype(np.float16)

        # 6) Anti-wear: protect confident foreground -> alpha=1 (but never on bg2d)
        if fg_margin > 0:
            dev_w = max_abs3(W - BgW)
            dev_b = max_abs3(B - BgB)
            fg_conf = (dev_w > np.float16(fg_margin)) | (dev_b > np.float16(fg_margin))
            alpha = np.where((~bg2d) & fg_conf, np.float16(1.0), alpha).astype(np.float16)

        # 7) Enforce background alpha=0
        alpha_u8 = alpha_to_u8(alpha)
        alpha_u8[bg2d] = 0

        # ---- Debug mode: ALWAYS return stats instead of PNG ----
        bg_ratio = float(np.count_nonzero(bg2d)) / float(bg2d.size)
        nonzero_ratio = float(np.count_nonzero(alpha_u8)) / float(alpha_u8.size)

        if debug == 1:
            # Also show alpha quantiles to see "worn out" vs "mostly solid"
            a = alpha_u8.reshape(-1)
            q = np.quantile(a, [0.0, 0.1, 0.5, 0.9, 1.0]).tolist()
            return JSONResponse(status_code=200, content={
                "size": [w, h],
                "bg_ratio": bg_ratio,
                "alpha_nonzero_ratio": nonzero_ratio,
                "alpha_u8_quantiles": {"min,p10,median,p90,max": q},
                "tips": {
                    "if_empty_or_nearly_empty": "csökkentsd tol_relaxed (60→55) vagy black_tol (28→20)",
                    "if_worn_center": "növeld fg_margin (22→28) és/vagy csökkentsd noise_cut (0.012→0.008)",
                    "if_too_dark_or_black_patches": "emeld alpha_hard (0.97→0.985), és hagyd a bg_blur-t 8–12 között"
                }
            })

        alpha01 = u8_to_alpha01(alpha_u8)

        # 8) Color: avoid darkening.
        # If alpha is already near-opaque, use original WHITE color (or B); only unpremultiply near edges.
        a_safe = np.maximum(alpha01, np.float16(1e-3))[..., None]
        recovered = (B - (np.float16(1.0) - alpha01)[..., None] * BgB) / a_safe
        recovered = np.clip(recovered, 0.0, 255.0)

        # Where alpha is high -> keep original (prevents darkening)
        hard = (alpha01 >= np.float16(alpha_hard))
        out_rgb = recovered
        out_rgb[hard] = W[hard]  # use white version inside solid areas

        # Where alpha ~ 0 -> force RGB=0 to avoid colored dust
        out_rgb = np.where(alpha01[..., None] <= np.float16(noise_cut), 0.0, out_rgb).astype(np.uint8)

        out = Image.fromarray(out_rgb, mode="RGB").convert("RGBA")
        out.putalpha(Image.fromarray(alpha_u8, mode="L"))

        # cleanup
        del W_key_u8, W_u8, B_u8, BgW_u8, BgB_u8, W, B, BgW, BgB, diff32, pix_dist, bgdiff32, bg_dist, alpha, alpha_u8
        gc.collect()

        return StreamingResponse(io.BytesIO(png_bytes(out)), media_type="image/png")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Processing failed: {e}"})
