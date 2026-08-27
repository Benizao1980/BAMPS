#!/usr/bin/env python3
"""
tune_model.py

Hyperparameter tuning for BAMPS.

Supports:
- regression: predict MIC (optionally on log2 scale)
- classification: predict S/I/R (requires categorical labels in MIC file or pre-derived label columns)

Typical usage:
python scripts/tune_model.py \
  --feature-table outputs/amrfinder/amr_presence_absence.tsv \
  --mic-file data/phenotypes.tsv \
  --task regression \
  --classifier xgb \
  --antibiotics drug_a drug_b \
  --log2 \
  --n-iter 80 --cv 5 --n-jobs 8 \
  --outdir outputs/tuning/example_xgb_log2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.utils.validation import check_is_fitted
import joblib

# Silence the XGBoost glibc warning spam on older HPC nodes
warnings.filterwarnings("ignore", message=".*old version of glibc.*", category=FutureWarning)

# Optional imports
try:
    import xgboost as xgb  # type: ignore
except Exception:
    xgb = None

try:
    import lightgbm as lgb  # type: ignore
except Exception:
    lgb = None


# -----------------------------
# Helpers
# -----------------------------

def read_table(path: Path) -> pd.DataFrame:
    """
    Read a delimited file, not trusting extension.
    Handles CSV/TSV with delimiter inference.
    """
    try:
        return pd.read_csv(path, sep=None, engine="python", dtype=str)
    except Exception:
        try:
            return pd.read_csv(path, sep="\t", dtype=str)
        except Exception:
            return pd.read_csv(path, sep=",", dtype=str)


def strip_bom_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lstrip("\ufeff") for c in df.columns]
    return df


def norm_id_series(s: pd.Series, dot_suffix_to_underscore: bool = False) -> pd.Series:
    """
    Normalise sample IDs for joining.
    - strip whitespace
    - optionally convert final ".<digits>" to "_<digits>"
    """
    out = s.astype(str).str.strip()
    if dot_suffix_to_underscore:
        out = out.str.replace(r"\.(\d+)$", r"_\1", regex=True)
    return out


def clean_antibiotic(x: str) -> str:
    return str(x).strip().lower().replace(" ", "_")


def harmonise_ids(s: pd.Series) -> pd.Series:
    """Harmonise common sample-ID quirks across BAMPS inputs.

    - strips whitespace
    - converts trailing '.<digits>' to '_<digits>' (e.g. 'ABC.4' -> 'ABC_4')
    """
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\.(\d+)$", r"_\1", regex=True)
    return s


def to_numeric(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def pick_shared_id_col(
    feat_cols: List[str],
    mic_cols: List[str],
    prefer: List[str] = ["sample", "id", "ID", "Sample", "SAMPLE"],
) -> Optional[Tuple[str, str]]:
    """
    Return (feat_id_col, mic_id_col) using preferred names if present in BOTH.
    If not found, try any exact overlap between columns.
    """
    feat_set = set(feat_cols)
    mic_set = set(mic_cols)

    for cand in prefer:
        if cand in feat_set and cand in mic_set:
            return cand, cand

    overlap = [c for c in feat_cols if c in mic_set]
    if overlap:
        c = overlap[0]
        return c, c
    return None


def dilution_step_error_log2(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Metrics in log2 units (dilution steps).
    """
    dif = y_pred - y_true
    mae = float(np.nanmean(np.abs(dif)))
    rmse = float(np.sqrt(np.nanmean(dif ** 2)))
    within1 = float(np.nanmean(np.abs(dif) <= 1.0))
    within2 = float(np.nanmean(np.abs(dif) <= 2.0))
    return {"mae_log2": mae, "rmse_log2": rmse, "within1": within1, "within2": within2}


# -----------------------------
# Models + search spaces
# -----------------------------

def build_estimator_and_space(
    classifier: str,
    task: str,
    random_state: int,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Returns (estimator, param_distributions).
    """
    classifier = classifier.lower()
    task = task.lower()

    if task not in {"regression", "classification"}:
        raise ValueError("task must be regression or classification")

    # A small but useful baseline: Ridge / LogisticRegression with scaling
    if classifier in {"ridge", "linear"}:
        if task == "regression":
            est = Pipeline([
                ("scaler", StandardScaler(with_mean=False)),
                ("model", Ridge(random_state=random_state)),
            ])
            space = {
                "model__alpha": np.logspace(-4, 4, 50),
            }
            return est, space
        else:
            est = Pipeline([
                ("scaler", StandardScaler(with_mean=False)),
                ("model", LogisticRegression(
                    max_iter=5000,
                    solver="liblinear",
                    random_state=random_state,
                )),
            ])
            space = {
                "model__C": np.logspace(-4, 4, 50),
                "model__penalty": ["l1", "l2"],
            }
            return est, space

    if classifier in {"rf", "random_forest"}:
        if task == "regression":
            est = RandomForestRegressor(
                n_estimators=500,
                random_state=random_state,
                n_jobs=1,
            )
            space = {
                "n_estimators": [300, 500, 800, 1200],
                "max_depth": [None, 5, 10, 20, 40],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None],
            }
            return est, space
        else:
            est = RandomForestClassifier(
                n_estimators=500,
                random_state=random_state,
                n_jobs=1,
            )
            space = {
                "n_estimators": [300, 500, 800, 1200],
                "max_depth": [None, 5, 10, 20, 40],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None],
                "class_weight": [None, "balanced"],
            }
            return est, space

    if classifier in {"xgb", "xgboost"}:
        if xgb is None:
            raise ImportError("xgboost is not installed/available in this environment.")
        if task == "regression":
            est = xgb.XGBRegressor(
                objective="reg:squarederror",
                random_state=random_state,
                n_estimators=600,
                n_jobs=1,
            )
            space = {
                "n_estimators": [300, 600, 900, 1400],
                "max_depth": [2, 3, 4, 5, 6, 8],
                "learning_rate": np.linspace(0.01, 0.3, 30),
                "subsample": np.linspace(0.5, 1.0, 6),
                "colsample_bytree": np.linspace(0.5, 1.0, 6),
                "min_child_weight": [1, 2, 5, 10],
                "reg_alpha": np.logspace(-6, 1, 20),
                "reg_lambda": np.logspace(-3, 2, 20),
                "gamma": [0, 0.01, 0.1, 0.5, 1.0],
            }
            return est, space
        else:
            est = xgb.XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=random_state,
                n_estimators=600,
                n_jobs=1,
            )
            space = {
                "n_estimators": [300, 600, 900, 1400],
                "max_depth": [2, 3, 4, 5, 6, 8],
                "learning_rate": np.linspace(0.01, 0.3, 30),
                "subsample": np.linspace(0.5, 1.0, 6),
                "colsample_bytree": np.linspace(0.5, 1.0, 6),
                "min_child_weight": [1, 2, 5, 10],
                "reg_alpha": np.logspace(-6, 1, 20),
                "reg_lambda": np.logspace(-3, 2, 20),
                "gamma": [0, 0.01, 0.1, 0.5, 1.0],
            }
            return est, space

    if classifier in {"lgbm", "lightgbm"}:
        if lgb is None:
            raise ImportError("lightgbm is not installed/available in this environment.")
        if task == "regression":
            est = lgb.LGBMRegressor(
                random_state=random_state,
                n_estimators=1200,
                n_jobs=1,
            )
            space = {
                "n_estimators": [500, 900, 1200, 1800],
                "learning_rate": np.linspace(0.01, 0.25, 25),
                "num_leaves": [15, 31, 63, 127, 255],
                "max_depth": [-1, 3, 5, 8, 12],
                "min_child_samples": [5, 10, 20, 40, 80],
                "subsample": np.linspace(0.6, 1.0, 5),
                "colsample_bytree": np.linspace(0.6, 1.0, 5),
                "reg_alpha": np.logspace(-6, 1, 15),
                "reg_lambda": np.logspace(-3, 2, 15),
            }
            return est, space
        else:
            est = lgb.LGBMClassifier(
                random_state=random_state,
                n_estimators=1200,
                n_jobs=1,
            )
            space = {
                "n_estimators": [500, 900, 1200, 1800],
                "learning_rate": np.linspace(0.01, 0.25, 25),
                "num_leaves": [15, 31, 63, 127, 255],
                "max_depth": [-1, 3, 5, 8, 12],
                "min_child_samples": [5, 10, 20, 40, 80],
                "subsample": np.linspace(0.6, 1.0, 5),
                "colsample_bytree": np.linspace(0.6, 1.0, 5),
                "reg_alpha": np.logspace(-6, 1, 15),
                "reg_lambda": np.logspace(-3, 2, 15),
            }
            return est, space

    raise ValueError(f"Unknown classifier: {classifier}")


def scoring_for_task(task: str) -> str:
    # RandomizedSearchCV expects a single score string unless you do refit=dict.
    # We tune regression on neg MAE (log2 if you pass --log2).
    if task == "regression":
        return "neg_mean_absolute_error"
    # For multi-class classification, balanced_accuracy is a sane default.
    return "balanced_accuracy"


# -----------------------------
# Core tuning
# -----------------------------

@dataclass
class TuneRunSummary:
    antibiotic: str
    task: str
    classifier: str
    n_samples: int
    n_features: int
    cv: int
    n_iter: int
    best_score: float
    best_params: Dict[str, Any]
    # extra evaluation on full data with best estimator (not nested; just sanity)
    train_mae: Optional[float] = None
    train_rmse: Optional[float] = None
    train_r2: Optional[float] = None
    train_acc: Optional[float] = None
    train_bal_acc: Optional[float] = None
    train_f1_macro: Optional[float] = None


def tune_one(
    X: pd.DataFrame,
    y: pd.Series,
    antibiotic: str,
    task: str,
    classifier: str,
    random_state: int,
    n_iter: int,
    cv: int,
    n_jobs: int,
    outdir: Path,
) -> TuneRunSummary:
    est, space = build_estimator_and_space(classifier, task, random_state=random_state)

    if task == "classification":
        cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        cv_split = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    score = scoring_for_task(task)

    search = RandomizedSearchCV(
        estimator=est,
        param_distributions=space,
        n_iter=n_iter,
        scoring=score,
        cv=cv_split,
        random_state=random_state,
        n_jobs=n_jobs,
        refit=True,
        verbose=1,
    )

    search.fit(X, y)

    best = search.best_estimator_
    best_score = float(search.best_score_)
    best_params = search.best_params_

    # Write CV results per antibiotic
    cv_df = pd.DataFrame(search.cv_results_)
    cv_df.to_csv(outdir / "tuning" / f"cv_results_{clean_antibiotic(antibiotic)}.tsv", sep="\t", index=False)

    # Save model
    model_path = outdir / "models" / f"{clean_antibiotic(antibiotic)}__{task}__{classifier}.pkl"
    joblib.dump(best, model_path)

    # Basic train-set sanity metrics
    summary = TuneRunSummary(
        antibiotic=clean_antibiotic(antibiotic),
        task=task,
        classifier=classifier,
        n_samples=int(X.shape[0]),
        n_features=int(X.shape[1]),
        cv=cv,
        n_iter=n_iter,
        best_score=best_score,
        best_params={k: (v.item() if hasattr(v, "item") else v) for k, v in best_params.items()},
    )

    if task == "regression":
        pred = best.predict(X)
        summary.train_mae = float(mean_absolute_error(y, pred))
        summary.train_rmse = float(np.sqrt(mean_squared_error(y, pred)))
        summary.train_r2 = float(r2_score(y, pred))
    else:
        pred = best.predict(X)
        summary.train_acc = float(accuracy_score(y, pred))
        summary.train_bal_acc = float(balanced_accuracy_score(y, pred))
        summary.train_f1_macro = float(f1_score(y, pred, average="macro"))

    # Metadata YAML
    meta = {
        "summary": asdict(summary),
        "model_path": str(model_path),
    }
    with open(outdir / "models" / f"{clean_antibiotic(antibiotic)}__{task}__{classifier}__metadata.yaml", "w") as fh:
        yaml.safe_dump(meta, fh, sort_keys=False)

    return summary


# -----------------------------
# CLI
# -----------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tune BAMPS models with RandomizedSearchCV.")

    p.add_argument("--feature-table", required=True, help="Feature matrix (rows=samples, cols=features).")
    p.add_argument("--mic-file", required=True, help="Phenotype/MIC table.")
    p.add_argument("--id-col", default="sample", help="ID column name in feature table (default: sample).")
    p.add_argument(
        "--mic-id-col",
        default=None,
        help="ID column name in MIC table. If omitted, auto-detects a shared column (prefers sample).",
    )

    p.add_argument("--task", required=True, choices=["regression", "classification"])
    p.add_argument("--classifier", required=True, choices=["xgb", "lgbm", "rf", "ridge"])

    p.add_argument("--antibiotics", nargs="+", required=True, help="Antibiotic columns to tune (e.g., imipenem meropenem).")
    p.add_argument("--log2", action="store_true", help="For regression: model log2(MIC).")
    p.add_argument("--dot-suffix-to-underscore", action="store_true",
                   help="Normalise IDs by converting final .<digits> to _<digits> (e.g., X.4 -> X_4).")

    p.add_argument("--n-iter", type=int, default=80)
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--random-state", type=int, default=1)

    p.add_argument("--outdir", required=True, help="Output directory for tuning results and models.")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)
    ensure_dir(outdir / "tuning")
    ensure_dir(outdir / "models")

    feat = strip_bom_columns(read_table(Path(args.feature_table)))
    mic = strip_bom_columns(read_table(Path(args.mic_file)))

    # Auto-detect mic id col if not provided
    if args.mic_id_col is None:
        picked = pick_shared_id_col(list(feat.columns), list(mic.columns), prefer=["sample", "id", "ID", "Sample", "SAMPLE"])
        if picked is None:
            raise ValueError(
                "Could not auto-detect a shared ID column between feature table and MIC table.\n"
                f"Feature cols (first 20): {list(feat.columns)[:20]}\n"
                f"MIC cols (first 20): {list(mic.columns)[:20]}"
            )
        feat_id, mic_id = picked
        args.id_col = feat_id
        args.mic_id_col = mic_id

    if args.id_col not in feat.columns:
        raise ValueError(f"--id-col '{args.id_col}' not found in feature table columns: {list(feat.columns)[:20]}")
    if args.mic_id_col not in mic.columns:
        raise ValueError(f"--mic-id-col '{args.mic_id_col}' not found in MIC table columns: {list(mic.columns)[:20]}")

    # Normalise IDs for joining
    feat["_join_id"] = norm_id_series(feat[args.id_col], dot_suffix_to_underscore=args.dot_suffix_to_underscore)
    mic["_join_id"] = norm_id_series(mic[args.mic_id_col], dot_suffix_to_underscore=args.dot_suffix_to_underscore)

    # Index features by join id; drop ID column from X
    feat = feat.set_index("_join_id", drop=True)
    if args.id_col in feat.columns:
        feat = feat.drop(columns=[args.id_col])

    # Convert features to numeric (0/1 presence/absence etc.)
    X_all = feat.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # MIC table: keep join id + antibiotic columns
    mic = mic.set_index("_join_id", drop=True)

    # Summaries written at end
    summaries: List[TuneRunSummary] = []

    for ab in args.antibiotics:
        ab_col = ab
        # allow case-insensitive match
        if ab_col not in mic.columns:
            low_map = {c.lower(): c for c in mic.columns}
            if ab.lower() in low_map:
                ab_col = low_map[ab.lower()]
            else:
                raise ValueError(f"Antibiotic '{ab}' not found in MIC table columns (sample shown): {list(mic.columns)[:30]}")

        # Join
        y_raw = mic[ab_col].copy()

        # Keep overlap
        overlap = X_all.index.intersection(y_raw.index)
        if len(overlap) == 0:
            raise ValueError("No overlap between feature IDs and MIC IDs after harmonisation.")

        X = X_all.loc[overlap].copy()
        y = y_raw.loc[overlap].copy()

        if args.task == "regression":
            y_num = to_numeric(y)
            mask = y_num.notna() & (y_num > 0)
            X = X.loc[mask.values]
            y_num = y_num.loc[mask.values]
            if args.log2:
                y_num = np.log2(y_num.astype(float))
            print(f"\n=== Tuning {clean_antibiotic(ab)} (regression, {args.classifier}) n={X.shape[0]} ===")
            summ = tune_one(
                X=X,
                y=y_num,
                antibiotic=ab,
                task="regression",
                classifier=args.classifier,
                random_state=args.random_state,
                n_iter=args.n_iter,
                cv=args.cv,
                n_jobs=args.n_jobs,
                outdir=outdir,
            )
            summaries.append(summ)

        else:
            # classification expects categorical labels already (S/I/R)
            y_cat = y.astype(str).str.strip()
            # Accept common numeric encodings? Keep it strict for now.
            allowed = {"S", "I", "R"}
            mask = y_cat.isin(allowed)
            X = X.loc[mask.values]
            y_cat = y_cat.loc[mask.values]
            if y_cat.nunique() < 2:
                raise ValueError(f"{ab}: classification target has <2 classes after filtering (found {sorted(y_cat.unique())})")
            print(f"\n=== Tuning {clean_antibiotic(ab)} (classification, {args.classifier}) n={X.shape[0]} ===")
            summ = tune_one(
                X=X,
                y=y_cat,
                antibiotic=ab,
                task="classification",
                classifier=args.classifier,
                random_state=args.random_state,
                n_iter=args.n_iter,
                cv=args.cv,
                n_jobs=args.n_jobs,
                outdir=outdir,
            )
            summaries.append(summ)

    # Write summary table
    sum_df = pd.DataFrame([asdict(s) for s in summaries])
    sum_path = outdir / "tuning" / "summary.tsv"
    sum_df.to_csv(sum_path, sep="\t", index=False)

    print("\nDone.")
    print("Wrote:", sum_path)
    print("Models:", outdir / "models")
    print("CV results:", outdir / "tuning")


if __name__ == "__main__":
    main()
