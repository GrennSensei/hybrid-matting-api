import io
import gc
from collections import deque

import numpy as np
from PIL import Image, ImageFilter
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse, JSONResponse

app = FastAPI(title="Hybrid Matting API", version="1.1.0")

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

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "hybrid-matting-api",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "hybrid": "POST /matte/hybrid"
        },
        "note": "Open /docs to test. Root / returns JSON, not the PNG."
    }

@app.get("/health")
def health():
    return {"status": "ok"}

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

