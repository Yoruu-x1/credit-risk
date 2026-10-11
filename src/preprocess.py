"""Preprocessing baseline dan pemetaan fitur hasil encoding."""

from typing import Literal

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator
from src import config


def build_preprocessor(
    numeric_scaling: Literal["robust", "none"] = "robust",
    additional_numeric_columns: tuple[str, ...] = (),
    repayment_encoding: Literal["one_hot", "numeric"] = "one_hot",
) -> ColumnTransformer:
    """Build preprocessing with configurable repayment representation."""
    if numeric_scaling not in ("robust", "none"):
        raise ValueError("numeric_scaling harus 'robust' atau 'none'.")

    if repayment_encoding not in ("one_hot", "numeric"):
        raise ValueError(
            "repayment_encoding harus 'one_hot' atau 'numeric'."
        )

    if len(set(additional_numeric_columns)) != len(
        additional_numeric_columns
    ):
        raise ValueError("Nama fitur tambahan tidak boleh berulang.")

    if not set(additional_numeric_columns).issubset(
        config.AVAILABLE_ENGINEERED_COLUMNS
    ):
        raise ValueError("Fitur tambahan belum terdaftar di config.")

    numeric_columns = [
        *config.NUMERIC_COLUMNS,
        *additional_numeric_columns,
    ]
    categorical_columns = list(config.CATEGORICAL_COLUMNS)

    if repayment_encoding == "numeric":
        numeric_columns.extend(config.REPAYMENT_STATUS_COLUMNS)
        categorical_columns = list(
            config.DEMOGRAPHIC_CATEGORICAL_COLUMNS
        )

    numeric_transformer = (
        RobustScaler()
        if numeric_scaling == "robust"
        else "passthrough"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_columns),
            (
                "categorical",
                OneHotEncoder(
                    drop=None,
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    # Dicatat save_experiment dari template pipeline sebelum fit.
    preprocessor.categorical_encoding = (
        "one_hot"
        if repayment_encoding == "one_hot"
        else "one_hot_demographics_numeric_repayment"
    )

    return preprocessor


def get_feature_mapping(
    preprocessor: BaseEstimator,
) -> pd.DataFrame:
    """Map fitted output columns to source features and categories."""
    check_is_fitted(preprocessor)

    # Native preprocessor menyediakan mapping-nya sendiri.
    custom_mapper = getattr(
        preprocessor,
        "get_feature_mapping",
        None,
    )

    if callable(custom_mapper):
        return custom_mapper()

    # Jalur mapping OHE untuk LR, LightGBM, dan XGBoost.
    if not isinstance(preprocessor, ColumnTransformer):
        raise TypeError(
            "Jenis preprocessor belum didukung untuk feature mapping."
        )

    encoded_names = preprocessor.get_feature_names_out()
    rows: list[dict[str, object]] = []

    fitted_columns = {
        name: list(columns)
        for name, _, columns in preprocessor.transformers_
        if name in {"numeric", "categorical"}
    }

    numeric_slice = preprocessor.output_indices_["numeric"]

    for encoded_name, original_name in zip(
        encoded_names[numeric_slice],
        fitted_columns["numeric"],
        strict=True,
    ):
        rows.append(
            {
                "encoded_feature": encoded_name,
                "original_feature": original_name,
                "category_value": None,
                "category_label": None,
            }
        )

    encoder = preprocessor.named_transformers_["categorical"]
    categorical_slice = preprocessor.output_indices_["categorical"]
    categorical_names = iter(encoded_names[categorical_slice])

    for original_name, categories in zip(
        fitted_columns["categorical"],
        encoder.categories_,
        strict=True,
    ):
        labels = config.CATEGORY_LABELS.get(original_name, {})

        for category in categories:
            code = int(category)

            rows.append(
                {
                    "encoded_feature": next(categorical_names),
                    "original_feature": original_name,
                    "category_value": code,
                    "category_label": labels.get(
                        code,
                        f"Code {code}",
                    ),
                }
            )

    mapping = pd.DataFrame(rows)

    if len(mapping) != len(encoded_names):
        raise RuntimeError(
            "Jumlah fitur encoded dan mapping tidak cocok."
        )

    return mapping