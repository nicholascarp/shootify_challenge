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
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt


```

## Expected input layout
```bash
input_root/
  sample_0001/
    still_life.jpg
    still_mask.png        # optional (auto-generated if missing)
    on_model.jpg          # or on_model_orig.jpg
  sample_0002/
    still_life.jpg
    on_model.jpg
  ...
```
# Run - Batch an entire folder
```bash
python scripts/process_folder.py \
  --input-root  /path/to/my_input \
  --output-root /path/to/my_output \
  --device auto

```


# Run - Common Flags
```bash
Common flags:
--device {auto|cpu|cuda} (default: auto)
--min-degrade 4.5 minimum ΔE for synthetic degrade (challenge)
--use-ot enable optional OT (Sinkhorn) refinement (slower)
--no-debug don’t save intermediate masks

```

# Run - Outputs & metrics
```bash
Outputs & metrics
Per sample (inside /path/to/my_output/<sample_name>/):
on_model_mask.png — garment mask used
on_model_degraded.jpg — synthetic degrade (for the challenge)
corrected_on_model.jpg — final output
```
