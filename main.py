import io
import gc
from collections import deque
from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse, JSONResponse

app = FastAPI(title="Hybrid Matting API", version="1.0.0")

# =========================================================
# CONFIG (Render Free safe)
# =========================================================

MAX_SIDE_DEFAULT = 2048
BG_MAP_SIDE = 512


# =========================================================
# Helpers
# =========================================================

def resize_max(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / float(max(w, h))
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def pil_to_f16(img: Image.Image):
    return np.array(img, dtype=np.uint8).astype(np.float16)


def alpha_to_u8(alpha):
    return (np.clip(alpha, 0, 1) * 255).round().astype(np.uint8)


def u8_to_alpha(a):
    return a.astype(np.float16) / np.float16(255.0)


# =========================================================
# FLOODFILL (outer background remover)
# =========================================================

def is_near_white(px, tol):
    r, g, b = px
    return abs(r - 255) <= tol and abs(g - 255) <= tol and abs(b - 255) <= tol


def floodfill_white_border_mask(img: Image.Image, tol: int):
    w, h = img.size
    pix = img.load()

    mask = bytearray(w * h)
    visited = bytearray(w * h)
    q = deque()

    def idx(x, y):
        return y * w + x

    # seed border
    for x in range(w):
        if is_near_white(pix[x, 0], tol):
            q.append((x, 0))
        if is_near_white(pix[x, h - 1], tol):
            q.append((x, h - 1))

    for y in range(h):
        if is_near_white(pix[0, y], tol):
            q.append((0, y))
        if is_near_white(pix[w - 1, y], tol):
            q.append((w - 1, y))

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
                q.append((nx, ny))

    return mask


# =========================================================
# DIFFERENCE MATTING (inner detail)
# =========================================================

def build_bg_map(img, blur_radius):
    small = resize_max(img, BG_MAP_SIDE)
    small_blur = small.filter(ImageFilter.GaussianBlur(blur_radius))
    return small_blur.resize(img.size, Image.BILINEAR)


# =========================================================
# HYBRID ENDPOINT
# =========================================================

@app.post("/matte/hybrid")
async def hybrid_matte(
    white_file: UploadFile = File(...),
    black_file: UploadFile = File(...),
    max_side: int = Query(MAX_SIDE_DEFAULT, ge=512, le=6000),
    flood_tolerance: int = Query(60, ge=0, le=255),
    noise_cut: float = Query(0.02, ge=0, le=0.2),
    bg_blur: float = Query(18.0, ge=0.0, le=64.0),
):

    try:
        img_w = resize_max(Image.open(io.BytesIO(await white_file.read())).convert("RGB"), max_side)
        img_b = resize_max(Image.open(io.BytesIO(await black_file.read())).convert("RGB"), max_side)

        if img_w.size != img_b.size:
            return JSONResponse(status_code=400, content={"error": "Images must match size"})

        # ---------- FLOODFILL ----------
        flood_mask = floodfill_white_border_mask(img_w, flood_tolerance)

        # ---------- DIFFERENCE ----------
        W = pil_to_f16(img_w)
        B = pil_to_f16(img_b)

        BgW = pil_to_f16(build_bg_map(img_w, bg_blur))
        BgB = pil_to_f16(build_bg_map(img_b, bg_blur))

        denom = BgW - BgB
        denom = np.where(np.abs(denom) < 1.0, 1.0, denom)

        alpha_rgb = 1.0 - ((W - B) / denom)
        alpha = np.median(alpha_rgb, axis=2)
        alpha = np.clip(alpha, 0, 1)

        # Noise cut
        alpha = np.where(alpha < noise_cut, 0, alpha)

        # ---------- HYBRID MERGE ----------
        alpha_u8 = alpha_to_u8(alpha)

        w, h = img_w.size
        i = 0
        for y in range(h):
            for x in range(w):
                if flood_mask[i] == 1:
                    alpha_u8[y, x] = 0
                i += 1

        alpha = u8_to_alpha(alpha_u8)

        # ---------- COLOR RECOVERY ----------
        alpha_safe = np.maximum(alpha, 1e-3)[..., None]
        out_rgb = (B - (1 - alpha)[..., None] * BgB) / alpha_safe
        out_rgb = np.clip(out_rgb, 0, 255)

        out_rgb = np.where(alpha[..., None] <= noise_cut, 0, out_rgb)

        out = Image.fromarray(out_rgb.astype(np.uint8), mode="RGB").convert("RGBA")
        out.putalpha(Image.fromarray(alpha_to_u8(alpha), mode="L"))

        gc.collect()

        return StreamingResponse(io.BytesIO(out.tobytes()), media_type="image/png")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
