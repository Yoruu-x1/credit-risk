from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src import config
from src.experiments import build_cost_curve
from src.preprocess import get_feature_mapping


@dataclass
class StackingOOFResult:
    """Hasil OOF stacking pada training set."""

    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    feature_mapping: pd.DataFrame
    meta_coefficients: pd.DataFrame


def diagnose_oof(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Membandingkan kebijakan biaya dan F1 dari probabilitas OOF."""
    y = predictions["y_true"].to_numpy(dtype=int)

    probabilities = predictions[
        "probability_default"
    ].to_numpy(dtype=float)

    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Target harus memiliki kelas 0 dan 1.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilitas mengandung nilai non-finite.")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilitas harus berada pada rentang 0–1.")

    curve = build_cost_curve(y, probabilities).copy()

    positives = int(y.sum())
    curve["tp"] = positives - curve["fn"]
    curve["tn"] = len(y) - positives - curve["fp"]

    flagged = curve["tp"] + curve["fp"]

    curve["recall"] = curve["tp"] / positives

    curve["precision"] = (
        curve["tp"] / flagged.replace(0, np.nan)
    ).fillna(0.0)

    curve["f1"] = (
        2 * curve["tp"]
        / (2 * curve["tp"] + curve["fp"] + curve["fn"])
    )

    curve["flagged_pct"] = 100 * flagged / len(y)

    cost_policy = curve.sort_values(
        ["total_cost", "threshold"]
    ).iloc[0]

    f1_policy = curve.sort_values(
        ["f1", "total_cost", "threshold"],
        ascending=[False, True, True],
    ).iloc[0]

    rows = []

    for policy, selected in (
        ("cost", cost_policy),
        ("f1_diagnostic", f1_policy),
    ):
        rows.append(
            {
                "policy": policy,
                "threshold": float(selected["threshold"]),
                "cost_per_1000": float(
                    selected["cost_per_1000"]
                ),
                "total_cost": float(selected["total_cost"]),
                "f1": float(selected["f1"]),
                "recall_pct": 100 * float(selected["recall"]),
                "precision_pct": 100 * float(
                    selected["precision"]
                ),
                "flagged_pct": float(selected["flagged_pct"]),
                "fn": int(selected["fn"]),
                "fp": int(selected["fp"]),
                "roc_auc": float(
                    roc_auc_score(y, probabilities)
                ),
                "average_precision": float(
                    average_precision_score(y, probabilities)
                ),
                "brier_score": float(
                    brier_score_loss(y, probabilities)
                ),
            }
        )

    return pd.DataFrame(rows), curve


def run_stacking_oof(
    estimator: StackingClassifier,
    X: pd.DataFrame,
    y: pd.Series,
) -> StackingOOFResult:
    """Mengevaluasi seluruh stacking dengan outer CV pada training."""
    if not X.index.equals(y.index) or not X.index.is_unique:
        raise ValueError("Index X/y harus sama dan unik.")

    if set(y.unique()) != {0, 1}:
        raise ValueError("Target harus memiliki kelas 0 dan 1.")

    outer_cv = StratifiedKFold(
        n_splits=config.CV_FOLDS,
        shuffle=True,
        random_state=config.CV_RANDOM_STATE,
    )

    probabilities = np.full(len(y), np.nan)
    fold_numbers = np.zeros(len(y), dtype=int)

    fold_rows = []
    mappings = []
    coefficient_rows = []

    for fold, (train_indices, valid_indices) in enumerate(
        outer_cv.split(X, y),
        start=1,
    ):
        print(
            f"Stacking: outer fold {fold}/{config.CV_FOLDS}",
            flush=True,
        )

        started = perf_counter()

        fitted = clone(estimator)
        fitted.fit(
            X.iloc[train_indices],
            y.iloc[train_indices],
        )

        positive_column = np.flatnonzero(
            fitted.classes_ == 1
        ).item()

        fold_probabilities = fitted.predict_proba(
            X.iloc[valid_indices]
        )[:, positive_column]

        probabilities[valid_indices] = fold_probabilities
        fold_numbers[valid_indices] = fold

        fold_y = y.iloc[valid_indices]

        fold_rows.append(
            {
                "fold": fold,
                "train_rows": len(train_indices),
                "validation_rows": len(valid_indices),
                "roc_auc": roc_auc_score(
                    fold_y, fold_probabilities
                ),
                "average_precision": average_precision_score(
                    fold_y, fold_probabilities
                ),
                "seconds": perf_counter() - started,
            }
        )

        if fold == 1:
            for name, pipeline in fitted.named_estimators_.items():
                mapping = get_feature_mapping(
                    pipeline.named_steps["preprocess"]
                ).copy()

                mapping.insert(0, "base_model", name)
                mappings.append(mapping)

        for (name, _), coefficient in zip(
            fitted.estimators,
            fitted.final_estimator_.coef_[0],
            strict=True,
        ):
            coefficient_rows.append(
                {
                    "fold": fold,
                    "base_model": name,
                    "coefficient": float(coefficient),
                    "intercept": float(
                        fitted.final_estimator_.intercept_[0]
                    ),
                }
            )

        print(
            f"Fold {fold} selesai: "
            f"{fold_rows[-1]['seconds']:.1f} detik",
            flush=True,
        )

    if (
        not np.isfinite(probabilities).all()
        or (fold_numbers == 0).any()
    ):
        raise RuntimeError("Prediksi OOF tidak lengkap.")

    predictions = pd.DataFrame(
        {
            "source_row_index": X.index.to_numpy(),
            "fold": fold_numbers,
            "y_true": y.to_numpy(),
            "probability_default": probabilities,
        }
    )

    diagnose_oof(predictions)

    return StackingOOFResult(
        predictions=predictions,
        fold_metrics=pd.DataFrame(fold_rows),
        feature_mapping=pd.concat(
            mappings, ignore_index=True
        ),
        meta_coefficients=pd.DataFrame(coefficient_rows),
    )