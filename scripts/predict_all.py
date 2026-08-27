#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def find_models(
    model_dir: Path,
    task: str,
    feature_set: str | None = None,
    classifier: str | None = None,
) -> List[Tuple[str, Path]]:
    """
    Find model files using the BAMPS-ML naming scheme:

      <antibiotic>__<feature_set>__<task>__<classifier>.pkl

    Example:
      imipenem__amr__regression__xgb.pkl
      meropenem__gwas__regression__xgb.pkl
      imipenem__amr_plus_gwas__classification__rf.pkl
    """
    models: List[Tuple[str, Path]] = []

    for p in sorted(model_dir.glob("*.pkl")):
        stem = p.stem
        parts = stem.split("__")

        if len(parts) != 4:
            continue

        antibiotic, model_feature_set, model_task, model_classifier = parts

        if model_task != task:
            continue
        if feature_set is not None and model_feature_set != feature_set:
            continue
        if classifier is not None and model_classifier != classifier:
            continue

        models.append((antibiotic, p))

    return models


def _expand_mic_pred_files(outdir: Path, order: List[str] | None = None) -> List[Path]:
    """Return a list of preds_<ab>_mic.tsv paths (expanded from filesystem, not shell)."""
    found = {
        p.name.split("_")[1]: p
        for p in outdir.glob("preds_*_mic.tsv")
        if len(p.name.split("_")) >= 3
    }
    if not found:
        return []

    if order:
        files: List[Path] = []
        used = set()
        for ab in order:
            p = found.get(ab)
            if p is not None:
                files.append(p)
                used.add(ab)
        for ab, p in sorted(found.items()):
            if ab not in used:
                files.append(p)
        return files

    return [p for _ab, p in sorted(found.items())]


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Run predictions for all available models "
            "(classification and/or regression), and optionally plot a MIC panel."
        )
    )

    # prediction io
    ap.add_argument(
        "--feature-table",
        required=True,
        nargs="+",
        type=Path,
        help="One or more feature tables used for prediction.",
    )
    ap.add_argument(
        "--feature-prefix",
        nargs="*",
        default=None,
        help="Optional prefixes for each feature table (e.g. AMR GWAS).",
    )
    ap.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help="Directory containing models named <antibiotic>__<feature_set>__<task>__<classifier>.pkl",
    )
    ap.add_argument(
        "--outdir",
        required=True,
        type=Path,
        help="Where to write prediction TSVs.",
    )

    # what to run
    ap.add_argument(
        "--tasks",
        nargs="+",
        choices=["classification", "regression"],
        default=["classification", "regression"],
        help="Which tasks to run.",
    )
    ap.add_argument(
        "--antibiotics",
        nargs="*",
        default=None,
        help="Optional whitelist of antibiotics to run (names match model filenames).",
    )
    ap.add_argument(
        "--feature-set",
        default=None,
        help="Optional feature-set filter for model selection (e.g. amr, gwas, amr_plus_gwas).",
    )
    ap.add_argument(
        "--classifier",
        default=None,
        help="Optional classifier filter for model selection (e.g. xgb, rf, ridge).",
    )

    ap.add_argument(
        "--no-to-mic",
        action="store_true",
        help="For regression models, DO NOT convert log2(MIC) back to raw MIC.",
    )

    # script paths
    ap.add_argument(
        "--predict-script",
        type=Path,
        default=None,
        help="Path to predict.py (default: scripts/predict.py next to this file).",
    )
    ap.add_argument(
        "--plot-script",
        type=Path,
        default=None,
        help="Path to plot_predicted_mic_panel.py "
             "(default: scripts/plot_predicted_mic_panel.py next to this file).",
    )

    # optional: immediate panel plotting
    ap.add_argument(
        "--panel-out",
        type=Path,
        default=None,
        help="If set, render a MIC panel after regression predictions. "
             "Provide a base path (without extension).",
    )
    ap.add_argument(
        "--panel-order",
        nargs="+",
        default=None,
        help="Order of antibiotics along x-axis "
             "(e.g., ciprofloxacin imipenem meropenem colistin).",
    )
    ap.add_argument(
        "--panel-truth",
        type=Path,
        default=None,
        help="Path to truth MIC CSV to show overlay and compute metrics (optional).",
    )
    ap.add_argument(
        "--panel-truth-id-col",
        default=None,
        help="ID column in truth CSV (optional; default is plotter --id-col).",
    )
    ap.add_argument(
        "--panel-id-col",
        default="sample",
        help="ID column in prediction TSVs (default: sample).",
    )

    # optional: S/I/R confusion matrices after classification
    ap.add_argument(
        "--cm-outdir",
        type=Path,
        default=None,
        help="If set, render S/I/R confusion matrices after classification.",
    )
    ap.add_argument(
        "--cm-truth",
        type=Path,
        default=None,
        help="Truth CSV (S/I/R per drug or MICs). Required if --cm-outdir is set.",
    )
    ap.add_argument(
        "--cm-id-col",
        default="",
        help="ID column in truth CSV (default: first col).",
    )
    ap.add_argument(
        "--cm-order",
        nargs="+",
        default=None,
        help="Antibiotics to include in confusion evaluation "
             "(default: from model filenames).",
    )

    args = ap.parse_args()

    # validate multi-table inputs
    if args.feature_prefix and len(args.feature_prefix) != len(args.feature_table):
        sys.exit(
            "[ERROR] --feature-prefix must have the same number of values as --feature-table"
        )

    # resolve scripts
    scripts_dir = Path(__file__).resolve().parent
    predict_py = args.predict_script or (scripts_dir / "predict.py")
    plot_py = args.plot_script or (scripts_dir / "plot_predicted_mic_panel.py")
    cm_py = scripts_dir / "plot_SIR_confusion.py"

    if not predict_py.exists():
        sys.exit(f"[ERROR] Could not find predict.py at: {predict_py}")
    if args.panel_out and not plot_py.exists():
        sys.exit(f"[ERROR] --panel-out was given, but plot script not found at: {plot_py}")
    if args.cm_outdir and not cm_py.exists():
        sys.exit(f"[ERROR] --cm-outdir was given, but confusion script not found at: {cm_py}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    ran = 0
    failures = 0

    def run_one(ab: str, task: str):
        nonlocal ran, failures

        out = args.outdir / (
            f"preds_{ab}_SIR.tsv"
            if task == "classification"
            else f"preds_{ab}_{'log2mic' if args.no_to_mic else 'mic'}.tsv"
        )

        cmd = [
            sys.executable,
            str(predict_py),
            "--feature-table",
            *[str(p) for p in args.feature_table],
            "--model-dir",
            str(args.model_dir),
            "--antibiotic",
            ab,
            "--task",
            task,
            "--feature-set",
            str(args.feature_set),
            "--classifier",
            str(args.classifier),
            "--output",
            str(out),
        ]

        if args.feature_prefix:
            cmd += ["--feature-prefix", *args.feature_prefix]

        if task == "regression" and not args.no_to_mic:
            cmd.append("--to-mic")

        print(f"\n[RUN] {ab}  task={task}  → {out}")
        try:
            subprocess.run(cmd, check=True)
            ran += 1
        except subprocess.CalledProcessError as e:
            print(f"[WARN] Prediction failed for {ab} ({task}). Exit code {e.returncode}")
            failures += 1

    # gather models and run predictions
    for task in args.tasks:
        models = find_models(
            args.model_dir,
            task,
            feature_set=args.feature_set,
            classifier=args.classifier,
        )
        if not models:
            print(
                f"[INFO] No {task} models found in {args.model_dir}"
                f" (feature_set={args.feature_set}, classifier={args.classifier})"
            )
            continue

        to_run = models
        if args.antibiotics:
            wanted = {a.lower() for a in args.antibiotics}
            to_run = [(ab, p) for ab, p in models if ab.lower() in wanted]
            missing = wanted - {ab.lower() for ab, _ in models}
            if missing:
                print(f"[INFO] Skipping {task} for missing models: {', '.join(sorted(missing))}")

        for ab, _path in to_run:
            run_one(ab, task)

    print(f"\n[DONE] Successful predictions: {ran}  |  failures: {failures}")

    # Optionally plot S/I/R confusion matrices
    if args.cm_outdir:
        if not args.cm_truth:
            sys.exit("[ERROR] --cm-outdir requires --cm-truth (CSV with S/I/R or MICs).")

        args.cm_outdir.mkdir(parents=True, exist_ok=True)

        if args.cm_order:
            cm_order = args.cm_order
        else:
            cm_order = sorted(
                [
                    p.name.split("_")[1]
                    for p in args.outdir.glob("preds_*_SIR.tsv")
                ]
            )

        if not cm_order:
            print("[INFO] No SIR prediction files to evaluate; skipping confusion matrices.")
        else:
            cmd = [
                sys.executable,
                str(cm_py),
                "--pred-glob",
                str(args.outdir / "preds_*_SIR.tsv"),
                "--truth",
                str(args.cm_truth),
                "--outdir",
                str(args.cm_outdir),
                "--order",
                *cm_order,
            ]
            if args.cm_id_col:
                cmd += ["--id-col", args.cm_id_col]

            print(f"\n[PLOT] S/I/R confusion → {args.cm_outdir}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(
                    f"[WARN] Confusion plotting failed (exit {e.returncode}). "
                    f"Command was:\n  {' '.join(cmd)}"
                )

    # optionally plot MIC panel right away
    if args.panel_out:
        pred_files = _expand_mic_pred_files(args.outdir, order=args.panel_order)
        if not pred_files:
            print("[INFO] No MIC prediction files to plot; skipping panel.")
            sys.exit(0)

        cmd = [
            sys.executable,
            str(plot_py),
            "panel",
            "--pred",
            *[str(p) for p in pred_files],
            "--out",
            str(args.panel_out),
            "--id-col",
            str(args.panel_id_col),
        ]
        if args.panel_truth:
            cmd += ["--truth", str(args.panel_truth)]
            if args.panel_truth_id_col:
                cmd += ["--truth-id-col", str(args.panel_truth_id_col)]

        print(f"\n[PLOT] MIC panel → {args.panel_out}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(
                f"[WARN] Panel plotting failed (exit {e.returncode}). "
                f"Command was:\n  {' '.join(cmd)}"
            )
            sys.exit(1)


if __name__ == "__main__":
    main()