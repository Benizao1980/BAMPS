#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = None
    XGBRegressor = None

LOG = logging.getLogger("train_model")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python", dtype=str)


def harmonise_antibiotic_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def harmonise_id(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"\.(\d+)$", r"_\1", value)
    return value


def load_config(path: Optional[Path]) -> dict:
    if path is None or not path.exists():
        return {}
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def get_breakpoints_from_config(cfg: dict) -> Dict[str, dict]:
    bp = cfg.get("breakpoints_table", {})
    out: Dict[str, dict] = {}
    if isinstance(bp, dict):
        for k, v in bp.items():
            out[harmonise_antibiotic_name(k)] = v if isinstance(v, dict) else {}
    return out


def mic_to_sir(mic: pd.Series, bp: dict, mode: str = "sir") -> pd.Series:
    mic = pd.to_numeric(mic, errors="coerce")
    out = pd.Series(index=mic.index, dtype=object)
    if not bp:
        out[:] = np.nan
        return out

    s_thr = bp.get("S", None)
    i_thr = bp.get("I", None)
    r_thr = bp.get("R", None)
    if i_thr is None and s_thr is not None:
        i_thr = s_thr

    if mode == "binary":
        out[:] = "S"
        thr = r_thr if r_thr is not None else i_thr
        if thr is not None:
            out[mic > float(thr)] = "NS"
        else:
            out[:] = np.nan
    else:
        out[:] = "S"
        if i_thr is not None:
            out[mic > float(i_thr)] = "I"
        if r_thr is not None:
            out[mic > float(r_thr)] = "R"

    out[mic.isna()] = np.nan
    return out


def prepare_labels(
    mic: pd.Series,
    task: str,
    bp: dict,
    log2: bool = True,
    classification_mode: str = "sir",
) -> pd.Series:
    mic_num = pd.to_numeric(mic, errors="coerce")
    if task == "classification":
        return mic_to_sir(mic_num, bp, mode=classification_mode).dropna()
    mic_num = mic_num.astype(float)
    mic_num = mic_num.where(mic_num > 0)
    y = np.log2(mic_num) if log2 else mic_num
    return y.dropna()


def dilution_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    delta = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    return {
        "within_1_dilution": float(np.mean(delta <= 1)),
        "within_2_dilution": float(np.mean(delta <= 2)),
    }


def build_model(
    task: str,
    classifier: str,
    n_classes: Optional[int],
    random_state: int,
    n_jobs: int = 1,
    extra_params: Optional[dict] = None,
):
    extra_params = extra_params or {}

    if task == "classification":
        if classifier == "rf":
            return RandomForestClassifier(
                n_estimators=int(extra_params.get("n_estimators", 500)),
                max_depth=extra_params.get("max_depth", None),
                min_samples_split=int(extra_params.get("min_samples_split", 2)),
                min_samples_leaf=int(extra_params.get("min_samples_leaf", 1)),
                random_state=random_state,
                n_jobs=n_jobs,
            )
        if classifier == "ridge":
            return LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                max_iter=int(extra_params.get("max_iter", 2000)),
                n_jobs=n_jobs,
                random_state=random_state,
                multi_class="auto",
            )
        if classifier == "xgb":
            if XGBClassifier is None:
                raise ImportError("xgboost is not importable in this environment.")
            params = dict(
                objective="binary:logistic" if n_classes == 2 else "multi:softprob",
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=n_jobs,
                n_estimators=400,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
            )
            params.update(extra_params)
            if n_classes and n_classes > 2:
                params["num_class"] = n_classes
            else:
                params.pop("num_class", None)
            return XGBClassifier(**params)
        raise ValueError(f"Unsupported classifier for classification: {classifier}")

    if classifier == "rf":
        return RandomForestRegressor(
            n_estimators=int(extra_params.get("n_estimators", 800)),
            max_depth=extra_params.get("max_depth", None),
            min_samples_split=int(extra_params.get("min_samples_split", 2)),
            min_samples_leaf=int(extra_params.get("min_samples_leaf", 1)),
            random_state=random_state,
            n_jobs=n_jobs,
        )
    if classifier == "ridge":
        return Ridge(alpha=float(extra_params.get("alpha", 1.0)), random_state=random_state)
    if classifier == "xgb":
        if XGBRegressor is None:
            raise ImportError("xgboost is not importable in this environment.")
        params = dict(
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=random_state,
            n_jobs=n_jobs,
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        params.update(extra_params)
        return XGBRegressor(**params)
    raise ValueError(f"Unsupported classifier for regression: {classifier}")


def load_best_params(source: Optional[Path], antibiotic: str) -> dict:
    if source is None:
        return {}

    ab = harmonise_antibiotic_name(antibiotic)
    src = Path(source)

    def extract_params(d: dict) -> dict:
        if not isinstance(d, dict):
            return {}

        # Direct antibiotic-specific structure
        if isinstance(d.get(ab), dict):
            return d[ab]

        # Common wrappers
        for key in [
            "best_params",
            "params",
            "tuned_params",
            "model_params",
            "xgb_params",
            "estimator_params",
        ]:
            if isinstance(d.get(key), dict):
                return d[key]

        # Nested summary wrappers
        if isinstance(d.get("summary"), dict):
            for key in ["best_params", "params", "tuned_params"]:
                if isinstance(d["summary"].get(key), dict):
                    return d["summary"][key]

        # If this looks like a raw parameter dictionary, use it directly
        known_param_keys = {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "gamma",
            "reg_alpha",
            "reg_lambda",
            "alpha",
            "lambda",
            "eta",
        }
        if any(k in d for k in known_param_keys):
            return d

        return {}

    if src.is_dir():
        candidates = (
            list(src.glob(f"{ab}_params.json")) +
            list(src.glob(f"{ab}*.json")) +
            list(src.glob(f"{ab}__*__metadata.yaml")) +
            list(src.glob(f"{ab}__*.yaml")) +
            list(src.glob(f"{ab}*.yaml")) +
            list(src.glob(f"{ab}*.yml"))
        )

        for c in candidates:
            try:
                with open(c) as fh:
                    d = yaml.safe_load(fh)
            except Exception:
                continue

            params = extract_params(d)
            if params:
                return params

        return {}

    try:
        with open(src) as fh:
            d = yaml.safe_load(fh)
    except Exception:
        return {}

    return extract_params(d)


def infer_id_column(df: pd.DataFrame, requested: Optional[str] = None) -> str:
    candidates = []
    if requested:
        candidates.append(requested)
    candidates.extend(["sample", "id", "sample_id", "isolate", "genome", df.columns[0]])
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not infer ID column from columns: {list(df.columns)[:20]}")


def load_feature_table(
    path: Path,
    id_col: Optional[str],
    prefix: Optional[str],
    harmonise_ids: bool,
) -> Tuple[pd.DataFrame, dict]:
    df = read_table(path)
    real_id_col = infer_id_column(df, requested=id_col)
    df = df.set_index(real_id_col)
    if harmonise_ids:
        df.index = df.index.map(harmonise_id)
    original_cols = list(df.columns)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if prefix:
        df.columns = [f"{prefix.strip()}__{c}" for c in df.columns]
    meta = {
        "path": str(path),
        "id_col": real_id_col,
        "prefix": prefix or "",
        "n_samples": int(df.shape[0]),
        "n_features": int(df.shape[1]),
        "original_feature_count": len(original_cols),
    }
    return df, meta


def merge_feature_tables(tables: List[pd.DataFrame]) -> pd.DataFrame:
    if not tables:
        raise ValueError("No feature tables supplied.")
    shared = tables[0].index
    for t in tables[1:]:
        shared = shared.intersection(t.index)
    if len(shared) == 0:
        raise ValueError("No overlapping sample IDs across feature tables.")
    merged = pd.concat([t.loc[shared] for t in tables], axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    return merged


def build_feature_set_label(feature_metadata: List[dict]) -> str:
    labels = []
    for meta in feature_metadata:
        prefix = meta.get("prefix", "")
        labels.append(prefix.lower() if prefix else Path(meta["path"]).stem.lower())
    return "_plus_".join(labels)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    class_labels: List[str],
) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "n_test": int(len(y_true)),
    }
    if y_proba is not None:
        try:
            if len(class_labels) == 2 and y_proba.ndim == 2 and y_proba.shape[1] == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            elif len(class_labels) > 2 and y_proba.ndim == 2:
                metrics["roc_auc_ovr_macro"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                )
        except Exception:
            pass
    return metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    out = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "n_test": int(len(y_true)),
    }
    out.update(dilution_metrics(y_true, y_pred))
    return out


def bootstrap_evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str,
    reps: int,
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []
    n = len(y_true)
    for i in range(reps):
        idx = rng.integers(0, n, size=n)
        yt = np.asarray(y_true)[idx]
        yp = np.asarray(y_pred)[idx]
        if task == "classification":
            rows.append({
                "replicate": i + 1,
                "accuracy": float(accuracy_score(yt, yp)),
                "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
            })
        else:
            row = {"replicate": i + 1}
            row.update(regression_metrics(yt, yp))
            rows.append(row)
    return pd.DataFrame(rows)


def summarise_bootstrap(df: pd.DataFrame, antibiotic: str, task: str) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        if col == "replicate":
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.empty:
            continue
        rows.append({
            "antibiotic": antibiotic,
            "task": task,
            "metric": col,
            "mean": float(vals.mean()),
            "median": float(vals.median()),
            "ci95_low": float(vals.quantile(0.025)),
            "ci95_high": float(vals.quantile(0.975)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BAMPS-ML models from AMR/GWAS/hybrid feature tables.")
    parser.add_argument("--feature-table", required=True, type=Path, nargs="+", help="One or more feature tables.")
    parser.add_argument("--feature-prefix", nargs="*", default=None, help="Optional prefixes in same order as feature tables.")
    parser.add_argument("--id-col", default=None, help="Optional ID column for feature tables.")
    parser.add_argument("--mic-file", required=True, type=Path, help="Phenotype/MIC table.")
    parser.add_argument("--mic-id-col", default="sample", help="ID column in phenotype table.")
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    parser.add_argument("--classification-mode", choices=["sir", "binary"], default="sir")
    parser.add_argument("--classifier", choices=["xgb", "rf", "ridge"], default="xgb")
    parser.add_argument("--antibiotics", nargs="+", default=None, help="Antibiotics to model.")
    parser.add_argument("--all-antibiotics", action="store_true", help="Infer all antibiotics from phenotype table.")
    parser.add_argument("--log2", action="store_true", help="Model log2 MIC for regression.")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--plot-dir", required=True, type=Path)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--fit-full", action="store_true")
    parser.add_argument("--bootstrap-reps", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--tuned-params-dir", default=None, type=Path)
    parser.add_argument("--params-from", default=None, type=Path)
    parser.add_argument("--config", default=None, type=Path)
    parser.add_argument("--harmonise-ids", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    if not args.self_test and not args.fit_full:
        LOG.info("Neither --self-test nor --fit-full supplied; defaulting to both.")
        args.self_test = True
        args.fit_full = True

    ensure_dir(args.model_dir)
    ensure_dir(args.plot_dir)

    cfg = load_config(args.config)
    bp_table = get_breakpoints_from_config(cfg)
    tuned_source = args.params_from if args.params_from is not None else args.tuned_params_dir

    prefixes = args.feature_prefix
    if prefixes is not None and len(prefixes) not in (0, len(args.feature_table)):
        raise ValueError("--feature-prefix must be omitted or have same length as --feature-table.")
    if not prefixes:
        prefixes = [None] * len(args.feature_table)

    feature_tables = []
    feature_meta = []
    for ft_path, prefix in zip(args.feature_table, prefixes):
        df, meta = load_feature_table(ft_path, args.id_col, prefix, args.harmonise_ids)
        feature_tables.append(df)
        feature_meta.append(meta)

    X = merge_feature_tables(feature_tables)
    feature_set_label = build_feature_set_label(feature_meta)
    LOG.info("Merged feature matrix: %d samples x %d features", X.shape[0], X.shape[1])

    mic_df = read_table(args.mic_file)
    if "fasta_file" in mic_df.columns and args.mic_id_col not in mic_df.columns:
        mic_df[args.mic_id_col] = mic_df["fasta_file"].apply(lambda x: Path(str(x)).stem)
    if args.mic_id_col not in mic_df.columns:
        raise ValueError(f"--mic-id-col '{args.mic_id_col}' not found in MIC table.")
    mic_df = mic_df.set_index(args.mic_id_col)
    if args.harmonise_ids:
        mic_df.index = mic_df.index.map(harmonise_id)

    shared = X.index.intersection(mic_df.index)
    if len(shared) == 0:
        raise ValueError("No overlap between feature IDs and MIC IDs after harmonisation.")
    X = X.loc[shared]
    mic_df = mic_df.loc[shared]
    LOG.info("Shared samples with phenotype data: %d", len(shared))

    if args.all_antibiotics:
        reserved = {"sample", "sample_id", "id", "fasta_file"}
        antibiotics = [c for c in mic_df.columns if harmonise_antibiotic_name(c) not in reserved]
    elif args.antibiotics:
        antibiotics = args.antibiotics
    else:
        antibiotics = cfg.get("antibiotics", [])
    if not antibiotics:
        raise ValueError("No antibiotics specified.")

    run_name = args.run_name or f"{feature_set_label}_{args.task}_{args.classifier}"
    run_dir = args.plot_dir / run_name
    ensure_dir(run_dir)
    pd.Series(X.columns, name="feature").to_csv(run_dir / "feature_columns.tsv", sep="\t", index=False)

    all_metrics = []
    all_bootstrap = []
    all_bootstrap_summary = []

    for ab_raw in antibiotics:
        ab = harmonise_antibiotic_name(ab_raw)
        matching = [c for c in mic_df.columns if harmonise_antibiotic_name(c) == ab]
        if not matching:
            LOG.warning("Skipping %s (not found in phenotype table).", ab_raw)
            continue
        mic_col = matching[0]
        bp = bp_table.get(ab, {})
        y = prepare_labels(mic_df[mic_col], args.task, bp, args.log2, args.classification_mode)
        Xy = X.loc[y.index]
        y = y.loc[Xy.index]
        if len(y) < 10:
            LOG.warning("Skipping %s; too few labelled samples (%d).", ab, len(y))
            continue

        LOG.info("Training %s | task=%s | classifier=%s | feature_set=%s | n=%d", ab, args.task, args.classifier, feature_set_label, len(y))
        extra_params = load_best_params(tuned_source, ab)
        if tuned_source is not None and not extra_params:
            raise ValueError(
                f"--params-from/--tuned-params-dir was supplied, but no tuned parameters "
                f"were loaded for antibiotic '{ab}' from {tuned_source}"
            )
        if extra_params:
            LOG.info("Using tuned params for %s: %s", ab, list(extra_params.keys())[:10])

        class_labels = None
        y_model = y.copy()
        if args.task == "classification":
            class_labels = sorted(set(y.astype(str)))
            class_to_int = {c: i for i, c in enumerate(class_labels)}
            y_model = y.astype(str).map(class_to_int).astype(int)
            stratify = y_model
            n_classes = len(class_labels)
        else:
            y_model = pd.to_numeric(y, errors="coerce").astype(float)
            stratify = None
            n_classes = None

        if args.self_test:
            X_train, X_test, y_train, y_test = train_test_split(
                Xy, y_model, test_size=args.test_size, random_state=args.random_state, stratify=stratify
            )
            model = build_model(args.task, args.classifier, n_classes if args.task == "classification" else None, args.random_state, args.n_jobs, extra_params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if args.task == "classification":
                y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
                metrics = classification_metrics(y_test, y_pred, y_proba, class_labels)
                pred_df = pd.DataFrame({"sample": X_test.index, "y_true": y_test, "y_pred": y_pred})
                pred_df["y_true_label"] = pred_df["y_true"].map({i: c for i, c in enumerate(class_labels)})
                pred_df["y_pred_label"] = pred_df["y_pred"].map({i: c for i, c in enumerate(class_labels)})
                with open(run_dir / f"{ab}_classification_report.txt", "w") as fh:
                    fh.write(classification_report(y_test, y_pred, target_names=class_labels))
                    fh.write("\n")
                cm = confusion_matrix(y_test, y_pred)
                pd.DataFrame(cm, index=class_labels, columns=class_labels).to_csv(run_dir / f"{ab}_confusion_matrix.tsv", sep="\t")
            else:
                metrics = regression_metrics(y_test, y_pred)
                pred_df = pd.DataFrame({
                    "sample": X_test.index,
                    "y_true": y_test,
                    "y_pred": y_pred,
                    "abs_error": np.abs(np.asarray(y_test) - np.asarray(y_pred)),
                })

            metrics.update({
                "antibiotic": ab,
                "task": args.task,
                "classifier": args.classifier,
                "feature_set": feature_set_label,
                "mode": "self_test",
                "n_train": int(X_train.shape[0]),
                "n_test": int(X_test.shape[0]),
                "n_features": int(Xy.shape[1]),
                "run_name": run_name,
            })
            all_metrics.append(metrics)
            pred_df.to_csv(run_dir / f"{ab}_self_test_predictions.tsv", sep="\t", index=False)

            if args.bootstrap_reps > 0:
                boot = bootstrap_evaluate(np.asarray(y_test), np.asarray(y_pred), args.task, args.bootstrap_reps, args.random_state)
                boot["antibiotic"] = ab
                boot["task"] = args.task
                boot["classifier"] = args.classifier
                boot["feature_set"] = feature_set_label
                boot["mode"] = "self_test"
                boot.to_csv(run_dir / f"{ab}_bootstrap_metrics.tsv", sep="\t", index=False)
                all_bootstrap.append(boot)
                boot_summary = summarise_bootstrap(boot.drop(columns=["antibiotic", "task", "classifier", "feature_set", "mode"]), ab, args.task)
                if not boot_summary.empty:
                    boot_summary["classifier"] = args.classifier
                    boot_summary["feature_set"] = feature_set_label
                    boot_summary["mode"] = "self_test"
                    all_bootstrap_summary.append(boot_summary)

        if args.fit_full:
            model_full = build_model(args.task, args.classifier, n_classes if args.task == "classification" else None, args.random_state, args.n_jobs, extra_params)
            model_full.fit(Xy, y_model)
            model_prefix = f"{ab}__{feature_set_label}__{args.task}__{args.classifier}"
            model_path = args.model_dir / f"{model_prefix}.pkl"
            meta_path = args.model_dir / f"{model_prefix}.meta.yaml"
            joblib.dump(model_full, model_path)
            meta = {
                "run_name": run_name,
                "antibiotic": ab,
                "task": args.task,
                "classification_mode": args.classification_mode if args.task == "classification" else None,
                "classifier": args.classifier,
                "feature_set": feature_set_label,
                "feature_tables": feature_meta,
                "n_samples": int(Xy.shape[0]),
                "n_features": int(Xy.shape[1]),
                "log2": bool(args.log2),
                "tuned_params": extra_params,
                "model_path": str(model_path),
                "id_col": args.id_col,
                "mic_id_col": args.mic_id_col,
                "harmonise_ids": bool(args.harmonise_ids),
                "class_labels": class_labels,
                "breakpoints": bp,
            }
            with open(meta_path, "w") as fh:
                yaml.safe_dump(meta, fh)
            all_metrics.append({
                "antibiotic": ab,
                "task": args.task,
                "classifier": args.classifier,
                "feature_set": feature_set_label,
                "mode": "fit_full",
                "n_samples": int(Xy.shape[0]),
                "n_features": int(Xy.shape[1]),
                "run_name": run_name,
                "model_path": str(model_path),
            })

    if all_metrics:
        pd.DataFrame(all_metrics).to_csv(run_dir / "all_metrics.tsv", sep="\t", index=False)
    if all_bootstrap:
        pd.concat(all_bootstrap, axis=0, ignore_index=True).to_csv(run_dir / "all_bootstrap_metrics.tsv", sep="\t", index=False)
    if all_bootstrap_summary:
        pd.concat(all_bootstrap_summary, axis=0, ignore_index=True).to_csv(run_dir / "all_bootstrap_summary.tsv", sep="\t", index=False)

    run_metadata = {
        "run_name": run_name,
        "task": args.task,
        "classifier": args.classifier,
        "classification_mode": args.classification_mode,
        "feature_set": feature_set_label,
        "feature_tables": feature_meta,
        "antibiotics_requested": antibiotics,
        "self_test": bool(args.self_test),
        "fit_full": bool(args.fit_full),
        "bootstrap_reps": int(args.bootstrap_reps),
        "log2": bool(args.log2),
        "random_state": int(args.random_state),
        "n_jobs": int(args.n_jobs),
    }
    with open(run_dir / "run_metadata.yaml", "w") as fh:
        yaml.safe_dump(run_metadata, fh)

    LOG.info("Done. Outputs written to: %s", run_dir)


if __name__ == "__main__":
    main()
