# Shootify Color Corrector (Option A)

Deterministic, prints-aware color correction for on-model images using a still-life reference.  
Key features:
- Automatic **on-model garment mask** (person parsing + skin removal + color-prior + GrabCut).
- **Chroma-aware degradation** generator (for challenge reproducibility).
- **Reinhard a\*b\*** transfer with luminance preservation and delta cap.
- Optional OT blend (disabled by default for speed/stability).
- Runs on CPU or GPU.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
