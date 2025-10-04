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

## Expected input layout
input_root/
  sample_0001/
    still_life.jpg
    still_mask.png        # optional (auto-generated if missing)
    on_model.jpg          # or on_model_orig.jpg
  sample_0002/
    still_life.jpg
    on_model.jpg
  ...

# Run
python scripts/process_folder.py \
  --input-root /path/to/your/folder \
  --output-root /path/to/output \
  --device auto \
  --min-degrade 4.5
