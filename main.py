import io
import gc
from collections import deque

import numpy as np
from PIL import Image, ImageFilter
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse

app = FastAPI(title="Hybrid Matting API", version="1.2.0")

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


# =========================================================
# Floodfill border background mask (outer bg remover)
# Uses "near white" on the WHITE image only
# =========================================================

def is_near_white(px, tol: int) -> bool:
    r, g, b = px
    return abs(r - 255) <= tol and abs(g - 255) <= tol and abs(b - 255) <= tol

def floodfill_white_border_mask(img_rgb: Image.Image, tol: int) -> bytearray:
    w, h = img_rgb.size
    pix = img_rgb.load()

    mask = bytearray(w * h)     # 1 = background connected to border
    visited = bytearray(w * h)

    q = deque()

    def idx(x, y):
        return y * w + x

    # seed border pixels that look like white bg
    for x in range(w):
        if is_near_white(pix[x, 0], tol): q.append((x, 0))
        if is_near_white(pix[x, h - 1], tol): q.append((x, h - 1))
    for y in range(h):
        if is_near_white(pix[0, y], tol): q.append((0, y))
        if is_near_white(pix[w - 1, y], tol): q.append((w - 1, y))

    while q:
        x, y = q.popleft()
        i = idx(x, y)
        if visited[i]:
            continue
        visited[i] = 1

        if not is_near_white(pix[x, y], tol):
            continue

        mask[i] = 1

        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < w and 0 <= ny < h:
                ni = idx(nx, ny)
                if not visited[ni]:
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
    # Minimal UI like the previous difference-matting API
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
      Upload the <b>white background</b> and <b>black background</b> versions of the same artwork (pixel-aligned).
      <br/>
      <b>Hybrid logic:</b> border-connected background removed by floodfill + inner details refined by difference matting.
    </p>

    <form action="/matte/hybrid?max_side=2048&flood_tolerance=60&bg_blur=18&bg_map_side=512&noise_cut=0.02"
          method="post" enctype="multipart/form-data"
          style="padding:16px; border:1px solid #ddd; border-radius:10px;">
      <p><b>White image (near #FFFFFF):</b></p>
      <input type="file" name="white_file" accept="image/*" required />

      <p style="margin-top:14px;"><b>Black image (near #000000):</b></p>
      <input type="file" name="black_file" accept="image/*" required />

      <p style="margin-top:14px;">
        <button type="submit" style="padding:10px 14px; font-weight:bold; cursor:pointer;">
          Generate Transparent PNG
        </button>
      </p>
    </form>

    <hr style="margin: 22px 0; border: none; border-top: 1px solid #eee;" />

    <h3 style="margin-bottom:6px;">Quick tuning</h3>
    <ul style="margin-top:6px; color:#444;">
      <li><b>noise_cut</b> (0.01–0.03): lower keeps more delicate details; higher removes more dust.</li>
      <li><b>flood_tolerance</b> (50–85): higher removes more border background, but can eat very pale edge pixels.</li>
      <li><b>bg_blur</b> (12–24): helps with gradients; too high can soften alpha math.</li>
      <li><b>max_side</b> (1600–2048): Render Free safe. 3000 may OOM.</li>
    </ul>

    <p style="color:#777;">
      Tip: If details are missing, try <b>noise_cut=0.015</b> and <b>flood_tolerance=55</b>.
      If background dust remains, try <b>noise_cut=0.03</b> and <b>flood_tolerance=70</b>.
    </p>

    <p style="color:#888; font-size: 13px;">
      API: POST <code>/matte/hybrid</code> — OpenAPI: <code>/docs</code>
    </p>
  </body>
</html>
"""


@app.post("/matte/hybrid")
async def hybrid_matte(
    white_file: UploadFile = File(..., description="Artwork on (near) white background"),
    black_file: UploadFile = File(..., description="Artwork on (near) black background"),

    max_side: int = Query(MAX_SIDE_DEFAULT, ge=512, le=6000),

    # Floodfill (outer bg)
    flood_tolerance: int = Query(60, ge=0, le=255),

    # Difference matting (inner detail)
    bg_blur: float = Query(18.0, ge=0.0, le=64.0),
    bg_map_side: int = Query(BG_MAP_SIDE_DEFAULT, ge=128, le=1024),
    noise_cut: float = Query(0.02, ge=0.0, le=0.2),
):
    try:
        img_w = Image.open(io.BytesIO(await white_file.read())).convert("RGB")
        img_b = Image.open(io.BytesIO(await black_file.read())).convert("RGB")

        img_w = resize_max(img_w, max_side=max_side)
        img_b = resize_max(img_b, max_side=max_side)

        if img_w.size != img_b.size:
            return JSONResponse(status_code=400, content={"error": "Images must match size (pixel-aligned)."})

        # 1) Floodfill mask on WHITE image (outer bg removal)
        flood_mask = floodfill_white_border_mask(img_w, tol=flood_tolerance)

        # 2) Difference alpha (background-aware, low-res bg map)
        W = pil_to_f16(img_w)
        B = pil_to_f16(img_b)

        BgW = pil_to_f16(build_bg_map_lowres(img_w, blur_radius=bg_blur, bg_map_side=bg_map_side))
        BgB = pil_to_f16(build_bg_map_lowres(img_b, blur_radius=bg_blur, bg_map_side=bg_map_side))

        denom = (BgW - BgB)
        denom_safe = np.where(np.abs(denom) < 1.0, np.sign(denom) * 1.0, denom).astype(np.float16)

        alpha_rgb = np.float16(1.0) - ((W - B) / denom_safe)
        alpha = np.median(alpha_rgb, axis=2)
        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float16)

        # noise cut
        if noise_cut > 0:
            alpha = np.where(alpha < np.float16(noise_cut), np.float16(0.0), alpha).astype(np.float16)

        # 3) Hybrid merge: force outer bg to 0 alpha using flood mask
        alpha_u8 = alpha_to_u8(alpha)
        w, h = img_w.size
        i = 0
        for y in range(h):
            for x in range(w):
                if flood_mask[i] == 1:
                    alpha_u8[y, x] = 0
                i += 1

        alpha = u8_to_alpha01(alpha_u8)

        # 4) Color recovery (background-aware)
        a_safe = np.maximum(alpha, np.float16(1e-3))[..., None]
        out_rgb = (B - (np.float16(1.0) - alpha)[..., None] * BgB) / a_safe
        out_rgb = np.clip(out_rgb, 0.0, 255.0)
        out_rgb = np.where(alpha[..., None] <= np.float16(noise_cut), 0.0, out_rgb).astype(np.uint8)

        out = Image.fromarray(out_rgb, mode="RGB").convert("RGBA")
        out.putalpha(Image.fromarray(alpha_to_u8(alpha), mode="L"))

        # free big arrays early
        del W, B, BgW, BgB, denom, denom_safe, alpha_rgb, out_rgb
        gc.collect()

        return StreamingResponse(io.BytesIO(png_bytes(out)), media_type="image/png")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Processing failed: {e}"})

