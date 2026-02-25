# Hybrid Matting API

Hybrid background removal:
- Outer background: floodfill
- Inner details: difference matting

POST /matte/hybrid

Form-data:
- white_file
- black_file

Query params:
- max_side (default 2048)
- flood_tolerance (default 60)
- noise_cut (default 0.02)
- bg_blur (default 18)
