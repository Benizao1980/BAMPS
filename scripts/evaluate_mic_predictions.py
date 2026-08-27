#!/usr/bin/env python3
"""evaluate_mic_predictions.py

Evaluation suite for BAMPS MIC predictions.

Inputs:
- One prediction TSV per antibiotic (as produced by predict_all.py), e.g. preds/preds_amikacin_mic.tsv
- A truth MIC table (wide or long)

Outputs (to --outdir):
- metrics_per_antibiotic.tsv
- metrics_overall.tsv
- merged_long.tsv.gz (useful for downstream analysis)
- plots (optional, --make-plots):
    - scatter_<ab>.png/.svg : log2(pred) vs log2(truth)
    - residuals_<ab>.png/.svg : (log2(pred)-log2(truth)) vs log2(truth)

This script is deliberately robust to:
- BOM headers
- truth wide columns named *_mic
- sample IDs that differ only by a trailing '.<digits>' vs '_<digits>'

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_table(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=None, engine="python", dtype=str)
    except Exception:
        try:
            df = pd.read_csv(path, sep="\t", dtype=str)
        except Exception:
            df = pd.read_csv(path, sep=",", dtype=str)
    df.columns = [str(c).lstrip("\ufeff") for c in df.columns]
    return df


def _clean_ab(x: str) -> str:
    s = str(x).strip().lower().replace(" ", "_")
    if s.endswith("_mic"):
        s = s[:-4]
    return s


def _harmonise_id(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\.(\d+)$", r"_\1", regex=True)
    return s


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _log2_pos(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    x = x.where(x > 0)
    return np.log2(x)


def _pick_col(cols: Sequence[str], preferred: str, fallbacks: Sequence[str]) -> str:
    cols_set = set(cols)
    if preferred in cols_set:
        return preferred
    for c in fallbacks:
        if c in cols_set:
            return c
    return preferred


def _infer_antibiotic_from_filename(p: Path) -> str:
    ab = p.name
    for pat in ["preds_", "pred_", "prediction_"]:
        if ab.startswith(pat):
            ab = ab[len(pat):]
    ab = ab.replace("_mic.tsv", "").replace(".tsv", "").replace(".csv", "")
    return _clean_ab(ab)


def _standardise_pred(df: pd.DataFrame, antibiotic: str, id_col: str, pred_col: Optional[str]) -> pd.DataFrame:
    if id_col not in df.columns:
        raise ValueError(f"Prediction file missing id column '{id_col}'. Columns: {list(df.columns)[:20]}")
    if pred_col is None:
        cand = [c for c in df.columns if c.lower() in {"pred", "prediction", "pred_mic", "mic_pred", "y_pred"}]
        if cand:
            pred_col = cand[0]
        else:
            for c in df.columns:
                if c == id_col:
                    continue
                if _to_num(df[c]).notna().mean() > 0.2:
                    pred_col = c
                    break
    if pred_col is None or pred_col not in df.columns:
        raise ValueError(f"Could not infer prediction column for {antibiotic}. Pass --pred-col.")

    out = pd.DataFrame({
        "sample_id": _harmonise_id(df[id_col]),
        "antibiotic": _clean_ab(antibiotic),
        "pred": _to_num(df[pred_col]),
    })
    return out.dropna(subset=["pred"])


def _standardise_truth(
    truth: pd.DataFrame,
    id_col: str,
    known_antibiotics: Sequence[str],
    antibiotic_col: str = "antibiotic",
    mic_col: str = "mic",
) -> pd.DataFrame:
    if id_col not in truth.columns:
        raise ValueError(f"Truth file must include '{id_col}' column. Columns: {list(truth.columns)}")

    cols = set(truth.columns)
    if antibiotic_col in cols and mic_col in cols:
        out = truth[[id_col, antibiotic_col, mic_col]].copy()
        out = out.rename(columns={id_col: "sample_id", antibiotic_col: "antibiotic", mic_col: "truth"})
        out["sample_id"] = _harmonise_id(out["sample_id"])
        out["antibiotic"] = out["antibiotic"].map(_clean_ab)
        out["truth"] = _to_num(out["truth"])
        return out.dropna(subset=["truth"])

    ab_set = set(_clean_ab(a) for a in known_antibiotics)
    ab_cols = []
    for c in truth.columns:
        if c == id_col:
            continue
        if _clean_ab(c) in ab_set:
            ab_cols.append(c)

    if not ab_cols:
        raise ValueError(
            "Could not identify antibiotic columns in truth (wide format).\n"
            "If your truth columns are like amikacin_mic, that's supported.\n"
            f"Known antibiotics: {sorted(ab_set)}\n"
            f"Truth columns (first 30): {list(truth.columns)[:30]}"
        )

    melt = truth.melt(id_vars=[id_col], value_vars=ab_cols, var_name="antibiotic", value_name="truth")
    melt = melt.rename(columns={id_col: "sample_id"})
    melt["sample_id"] = _harmonise_id(melt["sample_id"])
    melt["antibiotic"] = melt["antibiotic"].map(_clean_ab)
    melt["truth"] = _to_num(melt["truth"])
    return melt.dropna(subset=["truth"])


def _regression_metrics(log2_pred: pd.Series, log2_truth: pd.Series) -> Dict[str, float]:
    dif = (log2_pred - log2_truth).dropna()
    if len(dif) == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan, "within1": np.nan, "within2": np.nan}
    mae = float(np.mean(np.abs(dif)))
    rmse = float(np.sqrt(np.mean(dif ** 2)))
    # r2 on log2
    y = log2_truth.loc[dif.index].astype(float).values
    yhat = log2_pred.loc[dif.index].astype(float).values
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    within1 = float(np.mean(np.abs(dif) <= 1.0))
    within2 = float(np.mean(np.abs(dif) <= 2.0))
    return {"n": int(len(dif)), "mae": mae, "rmse": rmse, "r2": r2, "within1": within1, "within2": within2}


def _plot_scatter(df: pd.DataFrame, out_prefix: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.scatter(df["log2_truth"], df["log2_pred"], s=12, alpha=0.6)
    # y=x
    lims = [
        np.nanmin([df["log2_truth"].min(), df["log2_pred"].min()]),
        np.nanmax([df["log2_truth"].max(), df["log2_pred"].max()]),
    ]
    ax.plot(lims, lims, linestyle="--", linewidth=1)
    ax.set_xlabel("Truth MIC (log2)")
    ax.set_ylabel("Predicted MIC (log2)")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(str(out_prefix) + ".png", dpi=200)
    fig.savefig(str(out_prefix) + ".svg")
    plt.close(fig)


def _plot_residuals(df: pd.DataFrame, out_prefix: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.scatter(df["log2_truth"], df["residual"], s=12, alpha=0.6)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Truth MIC (log2)")
    ax.set_ylabel("Residual (pred - truth, log2)")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(str(out_prefix) + ".png", dpi=200)
    fig.savefig(str(out_prefix) + ".svg")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate BAMPS MIC predictions.")
    ap.add_argument("--pred", nargs="+", required=True, help="Prediction TSVs (one per antibiotic)")
    ap.add_argument("--truth", required=True, help="Truth MIC table (CSV/TSV; wide or long)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--id-col", default="sample", help="ID column in prediction TSVs")
    ap.add_argument("--pred-col", default=None, help="Prediction column name (optional)")
    ap.add_argument("--truth-id-col", default="id", help="ID column in truth table")
    ap.add_argument("--truth-antibiotic-col", default="antibiotic", help="Antibiotic col (long truth)")
    ap.add_argument("--truth-mic-col", default="mic", help="MIC col (long truth)")
    ap.add_argument("--make-plots", action="store_true", help="Write scatter/residual plots per antibiotic")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pred_paths = [Path(p) for p in args.pred]
    pred_rows = []
    antibiotics = []
    for p in pred_paths:
        df = _read_table(p)
        pid = _pick_col(df.columns, args.id_col, ["sample", "sample_id", "id", "genome", "isolate"])
        ab = _infer_antibiotic_from_filename(p)
        antibiotics.append(ab)
        pred_rows.append(_standardise_pred(df, antibiotic=ab, id_col=pid, pred_col=args.pred_col))
    pred_all = pd.concat(pred_rows, ignore_index=True)

    truth = _read_table(Path(args.truth))
    tid = _pick_col(truth.columns, args.truth_id_col, ["id", "sample", "sample_id"])
    truth_std = _standardise_truth(
        truth,
        id_col=tid,
        known_antibiotics=antibiotics,
        antibiotic_col=args.truth_antibiotic_col,
        mic_col=args.truth_mic_col,
    )

    merged = pred_all.merge(truth_std, on=["sample_id", "antibiotic"], how="inner").dropna(subset=["pred", "truth"])
    if len(merged) == 0:
        raise SystemExit("No overlapping sample_id+antibiotic between predictions and truth. Check ID harmonisation and truth columns.")

    # log2 columns
    merged["log2_pred"] = _log2_pos(merged["pred"])
    merged["log2_truth"] = _log2_pos(merged["truth"])
    merged = merged.dropna(subset=["log2_pred", "log2_truth"])
    merged["residual"] = merged["log2_pred"] - merged["log2_truth"]

    merged.to_csv(outdir / "merged_long.tsv.gz", sep="\t", index=False, compression="gzip")

    rows = []
    for ab, sub in merged.groupby("antibiotic"):
        met = _regression_metrics(sub["log2_pred"], sub["log2_truth"])
        met["antibiotic"] = ab
        rows.append(met)

        if args.make_plots:
            _plot_scatter(sub, outdir / f"scatter_{ab}", f"{ab} (n={met['n']})")
            _plot_residuals(sub, outdir / f"residuals_{ab}", f"{ab} residuals")

    per_ab = pd.DataFrame(rows).sort_values("antibiotic")
    per_ab.to_csv(outdir / "metrics_per_antibiotic.tsv", sep="\t", index=False)

    overall = _regression_metrics(merged["log2_pred"], merged["log2_truth"])
    overall_df = pd.DataFrame([{**overall, "antibiotic": "ALL"}])
    overall_df.to_csv(outdir / "metrics_overall.tsv", sep="\t", index=False)

    print(f"Wrote: {outdir / 'metrics_per_antibiotic.tsv'}")
    print(f"Wrote: {outdir / 'metrics_overall.tsv'}")
    if args.make_plots:
        print(f"Plots in: {outdir}")


if __name__ == "__main__":
    main()
