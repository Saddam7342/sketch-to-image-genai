# Input sketches

| File | Subject | Why it is a useful test case |
|---|---|---|
| `01_futuristic_tower.png` | Futuristic skyscraper with twin towers and sky bridge | Architecture: long straight lines, perspective-free elevation, repetitive facade grid |
| `02_robot_character.png` | Waving robot character | Character: articulated limbs, small facial details (eyes, visor), pose must survive |
| `03_hover_car.png` | Hover car, side view | Vehicle/product: smooth curves, one closed silhouette, hover pads under the body |
| `04_desk_lamp.png` | Articulated desk lamp | Product design: thin parts, joints, cone shape, light rays (non-object strokes) |
| `05_floating_island.png` | Floating island with cabin, trees, waterfall | Scene: many small objects at different scales, sky elements (clouds, birds) |

## Provenance

These sketches were **generated procedurally** by [`scripts/make_sketches.py`](../scripts/make_sketches.py),
a small "pen model" that draws strokes with hand-like wobble, overshoot, double lines and pressure
variation on a textured, unevenly lit paper background (imitating a phone photo of a pencil sketch).
The drawing script was written with AI assistance (Claude) and reviewed by the author. No external images
were used, so there are no copyright or permission issues. They can be regenerated exactly with:

```bash
python scripts/make_sketches.py --out sketches
```

Hand-drawn sketches (paper + phone photo) can be added to this folder at any time. The preprocessing
stage (`src/sketch2image/preprocess.py`) cleans uneven lighting and paper texture automatically. Add a
subject line for each new sketch under `sketches:` in `configs/experiments.yaml`.
