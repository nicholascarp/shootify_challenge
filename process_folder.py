#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# scripts/process_folder.py
import argparse, csv
from pathlib import Path
import cv2, numpy as np
from tqdm import tqdm

from src.color_corrector import (
    PersonSeg, still_mask_autogen, build_onmodel_mask_with_fallback,
    ensure_visible_degrade, correct_one, masked_de2000
)

def find_image(path: Path, names):
    for n in names:
        p = path / n
        if p.exists(): return p
    return None

def process_sample(sample_dir: Path, out_dir: Path, seg: PersonSeg,
                   min_degrade=4.5, use_ot=False, save_debug=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    # locate files (flexible names)
    still_p  = find_image(sample_dir, ["still_life.jpg","still-life.jpg","still.jpg"])
    on_p     = find_image(sample_dir, ["on_model.jpg","on_model_orig.jpg","model.jpg","on.jpg"])
    sm_p     = find_image(sample_dir, ["still_mask.png","still-mask.png","mask.png"])

    if still_p is None or on_p is None:
        return {"sample": sample_dir.name, "status": "missing_inputs"}

    sl = cv2.imread(str(still_p))
    om = cv2.imread(str(on_p))
    if sl is None or om is None:
        return {"sample": sample_dir.name, "status": "read_error"}

    if sm_p is not None:
        sm = cv2.imread(str(sm_p), 0)
    else:
        sm = still_mask_autogen(sl)
        if save_debug:
            cv2.imwrite(str(out_dir/"still_mask_autogen.png"), sm)

    # on-model mask
    mm = build_onmodel_mask_with_fallback(om, sl, sm, seg)
    if save_debug:
        cv2.imwrite(str(out_dir/"on_model_mask.png"), mm)

    if (mm>0).sum() < 300:
        # not confident—skip correction
        cv2.imwrite(str(out_dir/"corrected_on_model.jpg"), om)
        return {"sample": sample_dir.name, "status": "mask_small", "mask_area": int((mm>0).sum())}

    # degrade (for challenge reproducibility); if you're running on real on-model (already degraded), you can skip
    dg = ensure_visible_degrade(om, mm, min_de=min_degrade, max_tries=8)
    cv2.imwrite(str(out_dir/"on_model_degraded.jpg"), dg)
    de_deg = masked_de2000(om, dg, mm)

    # correct
    cor = correct_one(sl, sm, dg, om, mm, use_ot=use_ot)
    cv2.imwrite(str(out_dir/"corrected_on_model.jpg"), cor)
    de_out = masked_de2000(sl, cor, sm)  # lower is better (distance to still-life)

    return {
        "sample": sample_dir.name,
        "status": "ok",
        "mask_area": int((mm>0).sum()),
        "degrade_de": de_deg,
        "postcorr_de": de_out
    }

def main():
    ap = argparse.ArgumentParser(description="Shootify Color Corrector (Option A)")
    ap.add_argument("--input-root", required=True, type=Path,
                    help="Folder with subfolders, each containing still & on-model images.")
    ap.add_argument("--output-root", required=True, type=Path,
                    help="Where to write results.")
    ap.add_argument("--device", default="auto", help="cuda|cpu|auto")
    ap.add_argument("--min-degrade", type=float, default=4.5)
    ap.add_argument("--use-ot", action="store_true", help="Enable optional OT blend (slower).")
    ap.add_argument("--no-debug", action="store_true", help="Do not save intermediate masks.")
    args = ap.parse_args()

    seg = PersonSeg(device=args.device)
    rows = []

    # process each subfolder that contains at least an image
    sample_dirs = [p for p in args.input_root.iterdir() if p.is_dir()]
    for sd in tqdm(sorted(sample_dirs), desc="Processing"):
        out_dir = args.output_root / sd.name
        r = process_sample(sd, out_dir, seg,
                           min_degrade=args.min_degrade,
                           use_ot=args.use_ot,
                           save_debug=not args.no_debug)
        rows.append(r)

    # write metrics
    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / "metrics.csv"
    keys = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print("Saved:", csv_path)

if __name__ == "__main__":
    main()


