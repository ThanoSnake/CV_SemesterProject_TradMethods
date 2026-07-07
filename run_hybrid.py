"""Run a U-Net + classical hybrid on one fold and score it against the U-Net baseline.

Two modes (see hybrid.py):
  --mode uncertainty : Hybrid #1, band = high-MC-uncertainty ring (needs the dropout net).
  --mode morph       : Hybrid #2, band = morphological ring (works on the dropout-free net).

For each mode you can run BOTH refiners (random walker + level set) so their results are
directly comparable; each refiner is written to its own JSON. Every JSON also carries the
*unrefined* U-Net scores per case, so the Dice/HD95 delta of the hybrid is explicit.

  # Hybrid #1 on the MC-Dropout net:
  python3 run_hybrid.py --tag mcdropout --fold 0 --mode uncertainty --refiner both \
      --dropout-p 0.4 --preprocessed-dir data/Task09_Spleen/preprocessed_A
  # Hybrid #2 on the weakened net:
  python3 run_hybrid.py --tag weak --fold 0 --mode morph --refiner both --dropout-p 0.0 \
      --preprocessed-dir data/Task09_Spleen/preprocessed_A
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "unets"))          # predictor / arch / mc_dropout

import argparse
import json
import time

import numpy as np

import config
import io_utils
import metrics as M
import hybrid as H
from predictor import UNetPredictor

_UNC_KEYS = {"mutual_info": "mutual_info", "entropy": "entropy", "fg_var": "fg_var"}


def _refiner_list(arg):
    if arg == "both":
        return ["rw", "ls"]
    return [arg]


def _delta(a, b):
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    return float(a - b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="base net tag (e.g. mcdropout | weak)")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--mode", choices=["uncertainty", "morph"], required=True)
    p.add_argument("--refiner", choices=["rw", "ls", "ls_cv", "both"], default="both")
    # band params
    p.add_argument("--uncertainty", choices=list(_UNC_KEYS), default="entropy")
    p.add_argument("--mc-samples", type=int, default=30)
    p.add_argument("--band-quantile", type=float, default=0.80)
    p.add_argument("--ring-radius", type=int, default=12)
    p.add_argument("--band-radius", type=int, default=6, help="morph-band half width")
    # refiner params
    p.add_argument("--rw-beta", type=float, default=130.0)
    p.add_argument("--ls-kind", choices=["gac", "cv"], default="gac")
    p.add_argument("--ls-iters", type=int, default=10)
    p.add_argument("--ls-smoothing", type=int, default=1)
    p.add_argument("--ls-balloon", type=float, default=1.0)
    # model / io
    p.add_argument("--dropout-p", type=float, default=0.4)
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--preprocessed-dir", default=None)
    p.add_argument("--splits", default=None)
    p.add_argument("--weights-dir", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--advanced", action="store_true", help="also HD95 / ASSD")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    if args.mode == "uncertainty" and args.dropout_p <= 0:
        raise SystemExit("--mode uncertainty needs a dropout net (--dropout-p > 0).")

    prep = args.preprocessed_dir or str(config.PREPROCESSED_DIR)
    splits = io_utils.load_splits(args.splits)
    weights_dir = args.weights_dir or str(config.RESULTS_DIR / "unets")
    out_dir = args.out_dir or str(config.RESULTS_DIR / "hybrid")
    os.makedirs(out_dir, exist_ok=True)
    ckpt = args.ckpt or os.path.join(weights_dir, f"{args.tag}_f{args.fold}_best.pth")
    if not os.path.exists(ckpt):
        raise SystemExit(f"checkpoint not found: {ckpt} (train the '{args.tag}' net first)")

    refiners = _refiner_list(args.refiner)
    predictor = UNetPredictor(ckpt, num_classes=args.num_classes, dropout_p=args.dropout_p)
    names = io_utils.fold_cases(splits, args.fold, "test")
    if args.limit:
        names = names[:args.limit]

    fg = config.FOREGROUND_LABEL
    base_scores, hyb_scores = {}, {r: {} for r in refiners}
    band_frac, times = {}, {r: [] for r in refiners}

    for name, image, label in io_utils.iter_cases(names, preprocessed_dir=prep):
        # 1) base U-Net prediction (+ band)
        if args.mode == "uncertainty":
            m = predictor.predict_mc(image, T=args.mc_samples, temperature=args.temperature,
                                     batch=args.batch)
            seg = m["seg"].astype(bool)
            unc = m[_UNC_KEYS[args.uncertainty]]
            band = H.uncertainty_band(seg, unc, quantile=args.band_quantile,
                                      ring_radius=args.ring_radius)
        else:
            seg = predictor.predict(image, batch=args.batch)["seg"].astype(bool)
            band = H.morph_band(seg, radius=args.band_radius)

        base_scores[name] = M.evaluate_case(seg, label, foreground_label=fg,
                                            spacing=None, advanced=args.advanced)
        band_frac[name] = float(band.mean())

        # 2) each refiner (band-anchored)
        for r in refiners:
            t0 = time.perf_counter()
            refined = H.refine_volume(
                image, seg, band, refiner=r, beta=args.rw_beta, ls_kind=args.ls_kind,
                iters=args.ls_iters, smoothing=args.ls_smoothing, balloon=args.ls_balloon)
            times[r].append(time.perf_counter() - t0)
            hyb_scores[r][name] = M.evaluate_case(refined, label, foreground_label=fg,
                                                  spacing=None, advanced=args.advanced)
        d0 = base_scores[name]["Dice"]
        deltas = " ".join(f"{r}:{_delta(hyb_scores[r][name]['Dice'], d0):+.4f}" for r in refiners)
        print(f"  {name:16s} base={d0:.4f}  band={band_frac[name]*100:5.2f}%  d[{deltas}]", flush=True)

    base_agg = M.aggregate(base_scores)["mean"]
    for r in refiners:
        agg = M.aggregate(hyb_scores[r])["mean"]
        per_case = {n: {**hyb_scores[r][n],
                        "Dice_base": base_scores[n]["Dice"],
                        "Dice_delta": _delta(hyb_scores[r][n]["Dice"], base_scores[n]["Dice"]),
                        "band_frac": band_frac[n]} for n in hyb_scores[r]}
        meta = {"hybrid": args.mode, "refiner": r, "base_tag": args.tag, "fold": args.fold,
                "track": "A", "uncertainty": args.uncertainty if args.mode == "uncertainty" else None,
                "band": ({"quantile": args.band_quantile, "ring_radius": args.ring_radius}
                         if args.mode == "uncertainty" else {"radius": args.band_radius}),
                "ls_kind": args.ls_kind if r != "rw" else None,
                "n_cases": len(per_case),
                "sec_per_case_mean": float(np.mean(times[r])) if times[r] else float("nan")}
        stem = f"hybrid_{args.mode}_{r}_{args.tag}_f{args.fold}"
        payload = {"meta": meta, "mean": agg, "base_mean": base_agg, "per_case": per_case}
        with open(os.path.join(out_dir, f"{stem}.json"), "w") as f:
            json.dump(payload, f, indent=2)
        dd = agg["Dice"]["mean"] - base_agg["Dice"]["mean"]
        print(f"== {stem} ==  Dice {agg['Dice']['mean']:.4f} vs base {base_agg['Dice']['mean']:.4f} "
              f"(delta {dd:+.4f})  -> {os.path.join(out_dir, stem + '.json')}")


if __name__ == "__main__":
    main()
