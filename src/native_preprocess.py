from collections.abc import Sequence
from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from src import config


class NativeCategoricalPreprocessor(
    TransformerMixin,
    BaseEstimator,
):
    """Preserve numeric features and stringify categorical codes."""

    categorical_encoding = "native_codes_as_strings"

    def __init__(
        self,
        categorical_columns: tuple[str, ...] = (
            config.CATEGORICAL_COLUMNS
        ),
    ) -> None:
        self.categorical_columns = categorical_columns

    def _validate_input(self, X: pd.DataFrame) -> None:
        """Require finite features and integer category codes."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Input harus berupa pandas DataFrame."
            )

        if X.columns.has_duplicates:
            raise ValueError(
                "Nama kolom tidak boleh duplikat."
            )

        if not set(config.FEATURE_COLUMNS).issubset(X.columns):
            raise ValueError("Fitur asli tidak lengkap.")

        extra = set(X.columns) - set(config.FEATURE_COLUMNS)

        if not extra.issubset(
            config.AVAILABLE_ENGINEERED_COLUMNS
        ):
            raise ValueError(
                "Kolom tambahan tidak terdaftar."
            )

        if not set(self.categorical_columns).issubset(X.columns):
            raise ValueError(
                "Kolom kategorikal tidak lengkap."
            )

        if not np.isfinite(
            X.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                "Input mengandung missing atau nilai non-finite."
            )

        codes = X[
            list(self.categorical_columns)
        ].to_numpy(dtype=float)

        if not np.equal(codes, np.trunc(codes)).all():
            raise ValueError(
                "Kode kategorikal harus berupa bilangan bulat."
            )

    def fit(
        self,
        X: pd.DataFrame,
        y: object = None,
    ) -> Self:
        """Validate schema without learning category statistics."""
        self._validate_input(X)

        self.feature_names_in_ = np.asarray(
            X.columns,
            dtype=object,
        )
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert category codes without changing numeric values."""
        check_is_fitted(self, "feature_names_in_")
        self._validate_input(X)

        if list(X.columns) != self.feature_names_in_.tolist():
            raise ValueError(
                "Urutan kolom berbeda dari saat fit."
            )

        output = X.copy()

        for column in self.categorical_columns:
            output[column] = (
                output[column]
                .astype("int64")
                .astype(str)
            )

        return output

    def get_feature_names_out(
        self,
        input_features: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Return unchanged feature names."""
        check_is_fitted(self, "feature_names_in_")

        if input_features is not None:
            if list(input_features) != self.feature_names_in_.tolist():
                raise ValueError(
                    "Nama fitur tidak sesuai input saat fit."
                )

        return self.feature_names_in_.copy()

    def get_feature_mapping(self) -> pd.DataFrame:
        """Map native columns to original or derived feature names."""
        names = self.get_feature_names_out()

        return pd.DataFrame(
            {
                "encoded_feature": names,
                "original_feature": names,
                "category_value": [None] * len(names),
                "category_label": [None] * len(names),
            }
        )