"""Evaluasi OOF baseline dan pencarian threshold berbasis biaya."""

"""Evaluasi OOF baseline dan pencatatan eksperimen."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src import config
from src.preprocess import get_feature_mapping


@dataclass(frozen=True)
class OOFResult:
    """Hasil pengembangan model tanpa evaluasi test."""

    summary: dict[str, float | int]
    fold_metrics: pd.DataFrame
    predictions: pd.DataFrame
    threshold_curve: pd.DataFrame
    feature_mapping: pd.DataFrame


def build_cost_curve(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Menghitung biaya pada seluruh batas keputusan probabilitas."""
    order = np.argsort(probabilities, kind="stable")
    sorted_probabilities = probabilities[order]
    sorted_targets = y_true[order]

    # Perubahan keputusan terjadi pada nilai probabilitas yang berbeda.
    thresholds = np.unique(
        np.r_[0.0, probabilities, 1.0]
    )

    # Jika ada batas tepat 0.5, gunakan batas ekuivalen di bawahnya.
    # Ini mempertahankan keputusan pada data OOF tanpa memilih 0.5.
    if np.any(thresholds == 0.5):
        below_half = sorted_probabilities[
            sorted_probabilities < 0.5
        ]
        lower = float(below_half[-1]) if len(below_half) else 0.0
        alternative = (lower + 0.5) / 2.0

        if alternative <= lower or alternative == 0.5:
            raise ValueError(
                "Batas ekuivalen untuk threshold 0.5 tidak dapat "
                "direpresentasikan dengan presisi numerik saat ini."
            )

        thresholds[thresholds == 0.5] = alternative
        thresholds = np.unique(thresholds)

    # Prediksi positif menggunakan probability >= threshold.
    below_count = np.searchsorted(
        sorted_probabilities,
        thresholds,
        side="left",
    )

    cumulative_defaults = np.r_[
        0,
        np.cumsum(sorted_targets),
    ]

    false_negatives = cumulative_defaults[below_count]
    total_non_defaults = int((y_true == 0).sum())

    false_positives = (
        total_non_defaults
        - (below_count - false_negatives)
    )

    costs = (
        config.FALSE_NEGATIVE_COST * false_negatives
        + config.FALSE_POSITIVE_COST * false_positives
    )

    return pd.DataFrame({
        "threshold": thresholds,
        "fn": false_negatives,
        "fp": false_positives,
        "total_cost": costs,
        "cost_per_1000": costs / len(y_true) * 1000,
    })


def run_oof_experiment(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> OOFResult:
    """Menilai pipeline dengan parameter tetap menggunakan OOF train."""
    if not X.index.equals(y.index):
        raise ValueError("Indeks fitur dan target harus selaras.")

    if set(y.unique()) != {0, 1}:
        raise ValueError("Target harus memiliki kelas 0 dan 1.")

    if y.value_counts().min() < config.CV_FOLDS:
        raise ValueError("Jumlah kelas tidak cukup untuk CV.")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError(
            "Baseline ini membutuhkan fitur tanpa missing atau infinity."
        )

    cv = StratifiedKFold(
        n_splits=config.CV_FOLDS,
        shuffle=True,
        random_state=config.CV_RANDOM_STATE,
    )

    oof_probabilities = np.full(len(X), np.nan)
    fold_ids = np.zeros(len(X), dtype=int)
    fold_rows: list[dict[str, float | int]] = []
    feature_mapping = pd.DataFrame()

    for fold, (train_positions, valid_positions) in enumerate(
        cv.split(X, y),
        start=1,
    ):
        fitted_pipeline = clone(pipeline)
        started = perf_counter()

        # Preprocessing dan model hanya belajar dari training fold.
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)

            fitted_pipeline.fit(
                X.iloc[train_positions],
                y.iloc[train_positions],
            )

        positive_column = list(fitted_pipeline.classes_).index(1)

        probabilities = fitted_pipeline.predict_proba(
            X.iloc[valid_positions]
        )[:, positive_column]

        if not np.isfinite(probabilities).all():
            raise ValueError("Probabilitas prediksi tidak valid.")

        if np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("Probabilitas harus berada pada rentang 0–1.")

        oof_probabilities[valid_positions] = probabilities
        fold_ids[valid_positions] = fold

        fold_rows.append({
            "fold": fold,
            "train_rows": len(train_positions),
            "validation_rows": len(valid_positions),
            "roc_auc": float(
                roc_auc_score(
                    y.iloc[valid_positions],
                    probabilities,
                )
            ),
            "average_precision": float(
                average_precision_score(
                    y.iloc[valid_positions],
                    probabilities,
                )
            ),
            "seconds": perf_counter() - started,
        })

        if fold == 1:
            feature_mapping = get_feature_mapping(
                fitted_pipeline.named_steps["preprocess"]
            )

    if np.isnan(oof_probabilities).any() or np.any(fold_ids == 0):
        raise RuntimeError("Prediksi OOF belum lengkap.")

    y_array = y.to_numpy(dtype=int)
    curve = build_cost_curve(y_array, oof_probabilities)

    # Jika biaya sama, pilih threshold lebih rendah.
    # Pada seri keputusan ini, threshold lebih rendah tidak menambah FN.
    best = curve.sort_values(
        ["total_cost", "threshold"]
    ).iloc[0]

    threshold = float(best["threshold"])
    predictions = (oof_probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_array,
        predictions,
        labels=[0, 1],
    ).ravel()

    fold_metrics = pd.DataFrame(fold_rows)
    total_cost = (
        config.FALSE_NEGATIVE_COST * fn
        + config.FALSE_POSITIVE_COST * fp
    )

    summary: dict[str, float | int] = {
        "train_rows": len(X),
        "threshold": threshold,
        "oof_roc_auc": float(
            roc_auc_score(y_array, oof_probabilities)
        ),
        "oof_average_precision": float(
            average_precision_score(y_array, oof_probabilities)
        ),
        "oof_f1_at_selected_threshold": float(
            f1_score(y_array, predictions, zero_division=0)
        ),
        "oof_total_cost": float(total_cost),
        "oof_cost_per_1000": float(total_cost / len(X) * 1000),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "cv_roc_auc_mean": float(fold_metrics["roc_auc"].mean()),
        "cv_roc_auc_std": float(fold_metrics["roc_auc"].std()),
        "cv_average_precision_mean": float(
            fold_metrics["average_precision"].mean()
        ),
        "cv_average_precision_std": float(
            fold_metrics["average_precision"].std()
        ),
    }

    prediction_table = pd.DataFrame({
        "source_row_index": X.index.to_numpy(),
        "fold": fold_ids,
        "y_true": y_array,
        "probability_default": oof_probabilities,
        "prediction": predictions,
    })

    return OOFResult(
        summary=summary,
        fold_metrics=fold_metrics,
        predictions=prediction_table,
        threshold_curve=curve,
        feature_mapping=feature_mapping,
    )

def summarize_cost_sensitivity(
    result: OOFResult,
    experiment_id: str,
) -> pd.DataFrame:
    """Membandingkan skenario biaya pada probabilitas OOF yang sama."""
    default_count = int(
        result.summary["tp"] + result.summary["fn"]
    )
    non_default_count = int(
        result.summary["tn"] + result.summary["fp"]
    )
    sample_count = int(result.summary["train_rows"])

    rows: list[dict[str, float | int | str]] = []

    for ratio in config.COST_RATIOS:
        fp_cost = config.FALSE_POSITIVE_COST
        fn_cost = ratio * fp_cost

        candidates = result.threshold_curve.copy()
        candidates["scenario_cost"] = (
            fn_cost * candidates["fn"]
            + fp_cost * candidates["fp"]
        )

        best = candidates.sort_values(
            ["scenario_cost", "threshold"]
        ).iloc[0]

        fn = int(best["fn"])
        fp = int(best["fp"])
        tp = default_count - fn
        flagged_count = tp + fp

        all_non_default_cost = fn_cost * default_count
        all_high_risk_cost = fp_cost * non_default_count

        best_simple_cost = min(
            all_non_default_cost,
            all_high_risk_cost,
        )

        rows.append({
            "experiment_id": experiment_id,
            "fn_to_fp_ratio": ratio,
            "threshold": float(best["threshold"]),
            "fn": fn,
            "fp": fp,
            "cost_per_1000": (
                float(best["scenario_cost"])
                / sample_count * 1000
            ),
            "best_simple_policy": (
                "All non-default"
                if all_non_default_cost <= all_high_risk_cost
                else "All high-risk"
            ),
            "reduction_vs_best_simple_pct": (
                1 - float(best["scenario_cost"]) / best_simple_cost
            ) * 100,
            "recall_pct": tp / default_count * 100,
            "precision_pct": (
                tp / flagged_count * 100
                if flagged_count else 0.0
            ),
            "flagged_pct": flagged_count / sample_count * 100,
        })

    return pd.DataFrame(rows)

def save_experiment(
    result: OOFResult,
    pipeline: Pipeline,
    experiment_id: str,
    numeric_scaling: str,
) -> Path:
    """Menyimpan laporan dan metadata tanpa mengekspor model final."""
    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    run_dir = (
        config.EXPERIMENTS_DIR
        / experiment_id
        / run_id
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    sensitivity = summarize_cost_sensitivity(
        result,
        experiment_id,
    )

    summary = pd.DataFrame([{
        "experiment_id": experiment_id,
        "run_id": run_id,
        **result.summary,
    }])

    tables = {
        "summary": summary,
        "fold_metrics": result.fold_metrics,
        "oof_predictions": result.predictions,
        "threshold_curve": result.threshold_curve,
        "feature_mapping": result.feature_mapping,
        "cost_sensitivity": sensitivity,
    }

    for key, table in tables.items():
        table.to_csv(
            run_dir / config.EXPERIMENT_FILES[key],
            index=False,
        )

    model = pipeline.named_steps["model"]

    metadata = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "model_type": type(model).__name__,
        "model_parameters": model.get_params(deep=False),
        "dataset_filename": config.DATA_PATH.name,
        "dataset_sha256": hashlib.sha256(
            config.DATA_PATH.read_bytes()
        ).hexdigest(),
        "feature_columns": list(config.FEATURE_COLUMNS),
        "numeric_scaling": numeric_scaling,
        "categorical_encoding": "one_hot",
        "feature_engineering": False,
        "cv_folds": config.CV_FOLDS,
        "cv_random_state": config.CV_RANDOM_STATE,
        "split_random_state": config.RANDOM_STATE,
        "test_size": config.TEST_SIZE,
        "false_negative_cost": config.FALSE_NEGATIVE_COST,
        "false_positive_cost": config.FALSE_POSITIVE_COST,
        "cost_ratios": list(config.COST_RATIOS),
        "pr_metric_definition": "average_precision",
        "evaluation_scope": "train_oof_development",
        "versions": {
            package: version(package)
            for package in [
                "numpy",
                "pandas",
                "scikit-learn",
                "lightgbm",
                "xgboost",
            ]
        },
    }

    fe_step = pipeline.named_steps.get("feature_engineering")

    selected_features = (
        tuple(fe_step.features)
        if fe_step is not None
        else ()
    )

    metadata["feature_engineering"] = fe_step is not None

    metadata["feature_engineering_parameters"] = (
        fe_step.get_params(deep=False)
        if fe_step is not None
        else None
    )

    metadata["engineered_feature_definitions"] = {
        name: config.ENGINEERED_FEATURE_DEFINITIONS[name]
        for name in selected_features
    }

    preprocessing_columns = {
        name: list(columns)
        for name, _, columns in pipeline.named_steps["preprocess"].transformers
        if name in {"numeric", "categorical"}
    }

    metadata["model_numeric_columns"] = preprocessing_columns["numeric"]
    metadata["model_categorical_columns"] = preprocessing_columns["categorical"]

    metadata["categorical_encoding"] = getattr(
        pipeline.named_steps["preprocess"],
        "categorical_encoding",
        "one_hot",
    )

    with (
        run_dir / config.EXPERIMENT_FILES["metadata"]
    ).open("w", encoding="utf-8") as file:
        json.dump(
            _json_safe(metadata),
            file,
            indent=2,
            allow_nan=False,
        )

    return run_dir


def _json_safe(value: object) -> object:
    """Convert metadata values into strict JSON-compatible objects."""
    if isinstance(value, np.generic):
        return _json_safe(value.item())

    if isinstance(value, float) and not np.isfinite(value):
        return {"__non_finite_float__": str(value)}

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    raise TypeError(
        f"Unsupported metadata type: {type(value).__name__}"
    )