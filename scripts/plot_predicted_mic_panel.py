#!/usr/bin/env python3
"""plot_predicted_mic_panel.py

Robust MIC panel plotting for BAMPS-ML.

Key robustness features (to stop the exact errors you've hit):
- Reads CSV/TSV regardless of extension (delimiter sniffing)
- Strips UTF-8 BOM from headers (\ufeffid -> id)
- Harmonises sample IDs across truth/pred (e.g. 21063... .4 -> 21063..._4)
- Wide truth tables may use *_mic columns; we automatically strip that suffix so they match
  prediction file-derived antibiotic names (amikacin, ciprofloxacin, ...).
- Uses Q25/Q50/Q75 directly (no median±IQR/2 arithmetic that can create <=0 MIC and NaNs)

Outputs:
- <out>.png and <out>.svg (panel)
- optionally, <out>_lineages.* if lineage present and requested

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------
# IO + normalisation
# -------------------------

def _read_table(path: Path) -> pd.DataFrame:
    """Read a delimited file without trusting extension."""
    try:
        df = pd.read_csv(path, sep=None, engine="python", dtype=str)
    except Exception:
        try:
            df = pd.read_csv(path, sep="\t", dtype=str)
        except Exception:
            df = pd.read_csv(path, sep=",", dtype=str)
    # strip BOM from column names
    df.columns = [str(c).lstrip("\ufeff") for c in df.columns]
    return df


def _clean_ab(x: str) -> str:
    s = str(x).strip().lower().replace(" ", "_")
    # tolerate "*_mic" in either truth headers or long-format antibiotic names
    if s.endswith("_mic"):
        s = s[:-4]
    return s


def _harmonise_id(series: pd.Series) -> pd.Series:
    """Make truth/pred IDs comparable.

    - strip whitespace
    - convert trailing '.<digits>' to '_<digits>' (common in your validation MIC table)
    """
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\.(\d+)$", r"_\1", regex=True)
    return s


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _pick_col(cols: Sequence[str], preferred: str, fallbacks: Sequence[str]) -> str:
    cols_set = set(cols)
    if preferred in cols_set:
        return preferred
    for c in fallbacks:
        if c in cols_set:
            return c
    return preferred


def _standardise_pred(df: pd.DataFrame, antibiotic: str, id_col: str, pred_col: Optional[str]) -> pd.DataFrame:
    if id_col not in df.columns:
        raise ValueError(f"Prediction file missing id column '{id_col}'. Columns: {list(df.columns)[:20]}")

    # infer pred_col if not provided
    if pred_col is None:
        cand = [c for c in df.columns if c.lower() in {"pred", "prediction", "pred_mic", "mic_pred", "y_pred"}]
        if cand:
            pred_col = cand[0]
        else:
            # first numeric-ish column that's not id
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

    # optional lineage if present
    for cand in ["lineage", "ic", "IC", "clonal_group", "cluster", "ST", "cc"]:
        if cand in df.columns:
            out["lineage"] = df[cand].astype(str)
            break

    return out.dropna(subset=["pred"])


def _standardise_truth(
    truth: pd.DataFrame,
    id_col: str,
    known_antibiotics: Optional[Sequence[str]] = None,
    antibiotic_col: str = "antibiotic",
    mic_col: str = "mic",
) -> pd.DataFrame:
    if id_col not in truth.columns:
        raise ValueError(f"Truth file must include '{id_col}' column. Columns: {list(truth.columns)}")

    cols = set(truth.columns)

    # long format
    if antibiotic_col in cols and mic_col in cols:
        out = truth[[id_col, antibiotic_col, mic_col] + (["lineage"] if "lineage" in cols else [])].copy()
        out = out.rename(columns={id_col: "sample_id", antibiotic_col: "antibiotic", mic_col: "truth"})
        out["sample_id"] = _harmonise_id(out["sample_id"])
        out["antibiotic"] = out["antibiotic"].map(_clean_ab)
        out["truth"] = _to_num(out["truth"])
        return out.dropna(subset=["truth"])

    # wide format
    if known_antibiotics is None:
        ab_cols = [c for c in truth.columns if c != id_col and c.lower() not in {"lineage", "sample", "sample_id"}]
    else:
        ab_set = set(_clean_ab(a) for a in known_antibiotics)
        ab_cols = []
        for c in truth.columns:
            if c == id_col or c.lower() == "lineage":
                continue
            if _clean_ab(c) in ab_set:
                ab_cols.append(c)

    if not ab_cols:
        raise ValueError(
            "Truth file doesn't look like long format (missing 'antibiotic'+'mic') and I couldn't identify antibiotic columns in wide format.\n"
            "Tip: if your truth is wide and uses *_mic columns, that's supported; if still failing, check your --truth-id-col."
        )

    melt = truth.melt(
        id_vars=[id_col] + (["lineage"] if "lineage" in truth.columns else []),
        value_vars=ab_cols,
        var_name="antibiotic",
        value_name="truth",
    )
    melt = melt.rename(columns={id_col: "sample_id"})
    melt["sample_id"] = _harmonise_id(melt["sample_id"])
    melt["antibiotic"] = melt["antibiotic"].map(_clean_ab)
    melt["truth"] = _to_num(melt["truth"])
    return melt.dropna(subset=["truth"])


def _log2_pos(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    x = x.where(x > 0)
    return np.log2(x)


# -------------------------
# Plotting
# -------------------------

def make_mic_panel(
    pred_paths: Sequence[Path],
    out_prefix: Path,
    truth_path: Optional[Path] = None,
    id_col: str = "sample",
    pred_col: Optional[str] = None,
    truth_id_col: str = "id",
    truth_antibiotic_col: str = "antibiotic",
    truth_mic_col: str = "mic",
    lineage_col: Optional[str] = None,
    top_lineages: int = 6,
) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    pred_rows: List[pd.DataFrame] = []
    antibiotics: List[str] = []

    for p in pred_paths:
        df = _read_table(p)
        pred_id = _pick_col(df.columns, id_col, ["sample", "sample_id", "id", "genome", "isolate"])

        # infer antibiotic name from filename
        stem = p.name
        ab = stem
        for pat in ["preds_", "pred_", "prediction_"]:
            if ab.startswith(pat):
                ab = ab[len(pat):]
        ab = ab.replace("_mic.tsv", "").replace(".tsv", "").replace(".csv", "")
        antibiotics.append(_clean_ab(ab))

        pred_rows.append(_standardise_pred(df, antibiotic=ab, id_col=pred_id, pred_col=pred_col))

    pred_all = pd.concat(pred_rows, ignore_index=True)

    truth_std = None
    merged = None
    if truth_path is not None:
        truth = _read_table(truth_path)
        # truth_id_col might be missing due to BOM, but _read_table strips it already
        truth_id = _pick_col(truth.columns, truth_id_col, ["id", "sample", "sample_id"])
        truth_std = _standardise_truth(
            truth,
            id_col=truth_id,
            known_antibiotics=antibiotics,
            antibiotic_col=truth_antibiotic_col,
            mic_col=truth_mic_col,
        )

        merged = pred_all.merge(truth_std, on=["sample_id", "antibiotic"], how="inner")
        merged = merged.dropna(subset=["pred", "truth"])

    order = sorted(set(pred_all["antibiotic"]))

    # compute Q25/Q50/Q75 in *raw* mg/L then log2 transform
    def qstats(df: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q25, q50, q75 = [], [], []
        for ab in order:
            v = df.loc[df["antibiotic"] == ab, col].astype(float).values
            v = v[np.isfinite(v)]
            if len(v) == 0:
                q25.append(np.nan); q50.append(np.nan); q75.append(np.nan)
            else:
                q25.append(np.nanpercentile(v, 25))
                q50.append(np.nanpercentile(v, 50))
                q75.append(np.nanpercentile(v, 75))
        return np.array(q25), np.array(q50), np.array(q75)

    p25, p50, p75 = qstats(pred_all, "pred")
    t25 = t50 = t75 = None
    label_suffix = {ab: "" for ab in order}

    if merged is not None and len(merged):
        t25, t50, t75 = qstats(merged, "truth")
        # per-ab MAE/RMSE on log2
        for ab in order:
            sub = merged[merged["antibiotic"] == ab]
            lp = _log2_pos(sub["pred"].astype(float))
            lt = _log2_pos(sub["truth"].astype(float))
            dif = (lp - lt).dropna()
            if len(dif):
                mae = float(np.mean(np.abs(dif)))
                rmse = float(np.sqrt(np.mean(dif ** 2)))
                label_suffix[ab] = f" (n={len(dif)}, MAE={mae:.2f}, RMSE={rmse:.2f})"

    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(max(11, len(order) * 1.35), 4.8))

    # predicted
    p50l = _log2_pos(pd.Series(p50))
    p25l = _log2_pos(pd.Series(p25))
    p75l = _log2_pos(pd.Series(p75))
    ax.errorbar(
        x - 0.13,
        p50l,
        yerr=[p50l - p25l, p75l - p50l],
        fmt="o",
        capsize=3,
        label="Predicted (median ± IQR)",
    )

    if t25 is not None:
        t50l = _log2_pos(pd.Series(t50))
        t25l = _log2_pos(pd.Series(t25))
        t75l = _log2_pos(pd.Series(t75))
        ax.errorbar(
            x + 0.13,
            t50l,
            yerr=[t50l - t25l, t75l - t50l],
            fmt="s",
            capsize=3,
            label="Truth (median ± IQR)",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([ab + label_suffix[ab] for ab in order], rotation=25, ha="right")
    ax.set_ylabel("MIC (log2 mg/L)")
    ax.set_title("Predicted vs truth MIC per antibiotic (log2 scale)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend(frameon=False, loc="best")

    fig.tight_layout()
    fig.savefig(str(out_prefix) + ".png", dpi=200)
    fig.savefig(str(out_prefix) + ".svg")
    plt.close(fig)

    # optional lineage panel (MAE by lineage group)
    if merged is not None and len(merged):
        lc = lineage_col or ("lineage" if "lineage" in merged.columns else None)
        if lc is not None and lc in merged.columns:
            _make_lineage_panel(merged, out_prefix.with_name(out_prefix.name + "_lineages"), lc, top_lineages, order)


def _make_lineage_panel(merged: pd.DataFrame, out_prefix: Path, lineage_col: str, top_lineages: int, order: List[str]) -> None:
    df = merged.copy()
    df["lineage"] = df[lineage_col].astype(str)
    top = df["lineage"].value_counts().head(top_lineages).index.tolist()
    df["lineage_group"] = np.where(df["lineage"].isin(top), df["lineage"], "Other")

    df["log2_pred"] = _log2_pos(df["pred"].astype(float))
    df["log2_truth"] = _log2_pos(df["truth"].astype(float))
    df["abs_err"] = (df["log2_pred"] - df["log2_truth"]).abs()

    rows = []
    for ab in order:
        sub = df[df["antibiotic"] == ab]
        for g, ss in sub.groupby("lineage_group"):
            rows.append({"antibiotic": ab, "lineage_group": g, "mae": ss["abs_err"].mean(), "n": len(ss)})

    met = pd.DataFrame(rows)
    if met.empty:
        return

    fig, ax = plt.subplots(figsize=(max(11, len(order) * 1.35), 4.8))
    x = np.arange(len(order))
    groups = met["lineage_group"].unique().tolist()
    offsets = np.linspace(-0.25, 0.25, num=len(groups))

    for off, g in zip(offsets, groups):
        y = []
        for ab in order:
            v = met[(met["antibiotic"] == ab) & (met["lineage_group"] == g)]["mae"]
            y.append(float(v.iloc[0]) if len(v) else np.nan)
        ax.plot(x + off, y, marker="o", linestyle="-", label=g)

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=25, ha="right")
    ax.set_ylabel("MAE (log2 MIC)")
    ax.set_title("Lineage-stratified MAE by antibiotic (top lineages)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend(frameon=False, ncol=min(4, len(groups)), loc="best")

    fig.tight_layout()
    fig.savefig(str(out_prefix) + ".png", dpi=200)
    fig.savefig(str(out_prefix) + ".svg")
    plt.close(fig)


# -------------------------
# CLI
# -------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot BAMPS-ML MIC panels.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("panel", help="Predicted vs truth MIC panel")
    p1.add_argument("--pred", nargs="+", required=True, help="Prediction TSVs (one per antibiotic)")
    p1.add_argument("--out", required=True, help="Output prefix (no extension)")
    p1.add_argument("--truth", default=None, help="Truth table (CSV/TSV). Optional.")
    p1.add_argument("--id-col", default="sample", help="ID column in prediction TSVs")
    p1.add_argument("--pred-col", default=None, help="Prediction column in prediction TSVs (optional)")
    p1.add_argument("--truth-id-col", default="id", help="ID column in truth table (default: id)")
    p1.add_argument("--truth-antibiotic-col", default="antibiotic", help="Truth antibiotic column (long format)")
    p1.add_argument("--truth-mic-col", default="mic", help="Truth MIC column (long format)")
    p1.add_argument("--lineage-col", default=None, help="Column for lineage stratification (optional)")
    p1.add_argument("--top-lineages", type=int, default=6, help="Top N lineages to show")

    return p


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    if args.cmd == "panel":
        make_mic_panel(
            pred_paths=[Path(x) for x in args.pred],
            out_prefix=Path(args.out),
            truth_path=Path(args.truth) if args.truth else None,
            id_col=args.id_col,
            pred_col=args.pred_col,
            truth_id_col=args.truth_id_col,
            truth_antibiotic_col=args.truth_antibiotic_col,
            truth_mic_col=args.truth_mic_col,
            lineage_col=args.lineage_col,
            top_lineages=args.top_lineages,
        )
    else:
        raise SystemExit("Unknown command")


if __name__ == "__main__":
    main()
