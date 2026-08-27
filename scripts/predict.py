#!/usr/bin/env python3
"""
BAMPS :: Predictor
-----------------------------------
Apply a trained model (per antibiotic) to one or more feature matrices.

Examples:
  # S/I/R classification
  python scripts/predict.py \
    --feature-table outputs/amrfinder/amr_presence_absence.tsv \
    --feature-prefix AMR \
    --model-dir outputs/models \
    --antibiotic meropenem \
    --task classification \
    --feature-set amr \
    --classifier xgb \
    --output outputs/preds/preds_meropenem_SIR.tsv

  # MIC regression (back to raw mg/L)
  python scripts/predict.py \
    --feature-table outputs/amrfinder/amr_presence_absence.tsv \
    --feature-prefix AMR \
    --model-dir outputs/models \
    --antibiotic meropenem \
    --task regression \
    --feature-set amr \
    --classifier xgb \
    --to-mic \
    --output outputs/preds/preds_meropenem_mic.tsv

  # Hybrid prediction
  python scripts/predict.py \
    --feature-table outputs/amrfinder/validation/amr_presence_absence.norm.tsv \
                    GWAS_prep/gwas_features_validation_locus_presence_absence.tsv \
    --feature-prefix AMR GWAS \
    --model-dir outputs/models \
    --antibiotic imipenem \
    --task regression \
    --feature-set amr_plus_gwas \
    --classifier xgb \
    --to-mic \
    --output outputs/preds/preds_imipenem_mic.tsv
"""
from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


LOG = logging.getLogger("bamps_ml.predict")


def setup_logging():
    if not LOG.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--feature-table",
        required=True,
        nargs="+",
        type=Path,
        help="One or more TSV feature tables with samples as rows and features as columns (index col = sample).",
    )
    p.add_argument(
        "--feature-prefix",
        nargs="*",
        default=None,
        help="Optional prefixes for feature tables (e.g. AMR GWAS). Must match number of tables if provided.",
    )
    p.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help="Directory containing models named <antibiotic>__<feature_set>__<task>__<classifier>.pkl",
    )
    p.add_argument(
        "--antibiotic",
        required=True,
        help="Antibiotic name (must match model filename prefix).",
    )
    p.add_argument(
        "--task",
        choices=["classification", "regression"],
        default="classification",
        help="classification = S/I/R; regression = log2(MIC)",
    )
    p.add_argument(
        "--feature-set",
        required=True,
        help="Feature set name in the model filename (e.g. amr, gwas, amr_plus_gwas).",
    )
    p.add_argument(
        "--classifier",
        required=True,
        help="Classifier name in the model filename (e.g. xgb, rf, ridge).",
    )
    p.add_argument(
        "--to-mic",
        action="store_true",
        help="(regression only) exponentiate log2 predictions back to MIC (mg/L)",
    )
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output TSV path",
    )

    return p.parse_args()


def read_feature_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def apply_prefix(df: pd.DataFrame, prefix: Optional[str]) -> pd.DataFrame:
    if prefix is None or str(prefix).strip() == "":
        return df
    prefix = str(prefix).strip()
    out = df.copy()
    out.columns = [f"{prefix}__{c}" for c in out.columns]
    return out


def load_feature_tables(paths: List[Path], prefixes: Optional[List[str]]) -> pd.DataFrame:
    dfs = []

    if prefixes is not None and len(prefixes) != len(paths):
        raise ValueError(
            "--feature-prefix must have the same number of values as --feature-table"
        )

    for i, path in enumerate(paths):
        prefix = prefixes[i] if prefixes is not None else None
        df = read_feature_table(path)
        df = apply_prefix(df, prefix)
        dfs.append(df)

        LOG.info(
            "Loaded feature table: %s | samples=%d | features=%d%s",
            path,
            df.shape[0],
            df.shape[1],
            f" | prefix={prefix}" if prefix else "",
        )

    merged = pd.concat(dfs, axis=1, join="outer").fillna(0)

    if merged.columns.duplicated().any():
        dup_n = int(merged.columns.duplicated().sum())
        LOG.warning("Dropping %d duplicated feature columns after merge", dup_n)
        merged = merged.loc[:, ~merged.columns.duplicated()]

    LOG.info(
        "Merged prediction matrix: %d samples x %d features",
        merged.shape[0],
        merged.shape[1],
    )
    return merged


def get_expected_features(model) -> List[str]:
    expected = getattr(model, "feature_names_in_", None)

    if expected is None:
        try:
            expected = model.get_booster().feature_names
        except Exception:
            expected = None

    if expected is None:
        raise ValueError(
            "Could not determine expected feature names from model. "
            "Please retrain with a model that stores feature names."
        )

    return [str(c) for c in expected]


def load_and_align_features(
    feature_paths: List[Path],
    feature_prefixes: Optional[List[str]],
    model,
) -> pd.DataFrame:
    X = load_feature_tables(feature_paths, feature_prefixes)

    expected = get_expected_features(model)
    current = set(map(str, X.columns))

    missing = [c for c in expected if c not in current]
    extra = [c for c in X.columns if c not in set(expected)]

    if missing:
        missing_df = pd.DataFrame(0, index=X.index, columns=missing)
        X = pd.concat([X, missing_df], axis=1)

    if extra:
        X = X.drop(columns=extra, errors="ignore")

    X = X.reindex(columns=expected, fill_value=0)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    if missing:
        LOG.info(
            "Added %d missing training features as 0 (e.g. %s)",
            len(missing),
            ", ".join(missing[:5]),
        )
    if extra:
        LOG.info(
            "Dropped %d unseen prediction features (e.g. %s)",
            len(extra),
            ", ".join(extra[:5]),
        )

    return X


def main():
    args = parse_args()
    setup_logging()

    if args.feature_prefix is not None and len(args.feature_prefix) == 0:
        args.feature_prefix = None

    if args.feature_prefix and len(args.feature_prefix) != len(args.feature_table):
        raise ValueError(
            "--feature-prefix must have the same number of values as --feature-table"
        )

    model_path = args.model_dir / (
        f"{args.antibiotic}__{args.feature_set}__{args.task}__{args.classifier}.pkl"
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    LOG.info("Loading model: %s", model_path)

    with open(model_path, "rb") as fh:
        model = pickle.load(fh)

    X = load_and_align_features(args.feature_table, args.feature_prefix, model)

    if args.task == "classification":
        yhat = model.predict(X)

        if hasattr(yhat, "ndim") and yhat.ndim == 2:
            yhat = np.asarray(yhat).argmax(axis=1)

        yhat = np.asarray(yhat).astype(int)

        n_classes = getattr(model, "n_classes_", None)
        if n_classes is None:
            n_classes = int(yhat.max() + 1) if yhat.size else 1

        names_by_k = {
            1: ["S"],
            2: ["S", "R"],
            3: ["S", "I", "R"],
        }
        names = names_by_k.get(n_classes, ["S", "I", "R"][:n_classes])
        pred_labels = [names[i] if 0 <= i < len(names) else "S" for i in yhat]

        result_df = pd.DataFrame(
            {
                "sample": X.index,
                "prediction": pred_labels,
            }
        )

    else:
        log2_pred = np.asarray(model.predict(X), dtype=float)

        if args.to_mic:
            mic = np.power(2.0, log2_pred)
            result_df = pd.DataFrame(
                {
                    "sample": X.index,
                    "prediction": mic,
                    "prediction_log2": log2_pred,
                }
            )
        else:
            result_df = pd.DataFrame(
                {
                    "sample": X.index,
                    "prediction": log2_pred,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output, sep="\t", index=False)

    LOG.info("Predictions written → %s", args.output)


if __name__ == "__main__":
    main()