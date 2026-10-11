from collections.abc import Sequence
from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from src import config


class CreditFeatureEngineer(TransformerMixin, BaseEstimator):
    """Append history summaries without using the target."""

    def __init__(
        self,
        features: tuple[str, ...] = config.ENGINEERED_NUMERIC_COLUMNS,
    ) -> None:
        self.features = features

    def _validate_input(self, X: pd.DataFrame) -> None:
        """Require finite raw features without ID or target."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input harus berupa pandas DataFrame.")

        if X.columns.has_duplicates:
            raise ValueError("Nama kolom tidak boleh duplikat.")

        if set(X.columns) != set(config.FEATURE_COLUMNS):
            raise ValueError(
                "Input harus berisi fitur asli tanpa ID atau target."
            )

        if not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError(
                "Input mengandung missing atau nilai non-finite."
            )
        uses_ratios = bool(
            set(self.features) & set(config.FE_REFINEMENT_RATIO_COLUMNS)
        )

        if uses_ratios and X["LIMIT_BAL"].le(0).any():
            raise ValueError(
                "Fitur rasio memerlukan LIMIT_BAL lebih besar dari nol."
            )

    def fit(
        self,
        X: pd.DataFrame,
        y: object = None,
    ) -> Self:
        """Validate schema without learning statistics or using labels."""
        self._validate_input(X)

        if len(set(self.features)) != len(self.features):
            raise ValueError(
                "Nama fitur tambahan tidak boleh berulang."
            )

        if not set(self.features).issubset(
            config.AVAILABLE_ENGINEERED_COLUMNS
        ):
            raise ValueError(
                "Fitur tambahan belum terdaftar di config."
            )

        self.feature_names_in_ = np.asarray(
            X.columns,
            dtype=object,
        )
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute history features and preserve original inputs."""
        check_is_fitted(self, "feature_names_in_")
        self._validate_input(X)

        if list(X.columns) != self.feature_names_in_.tolist():
            raise ValueError(
                "Urutan kolom berbeda dari saat fit."
            )

        bills = X.loc[:, list(config.BILL_COLUMNS)]
        payments = X.loc[:, list(config.PAYMENT_COLUMNS)]
        repayment = X.loc[
            :,
            list(config.REPAYMENT_STATUS_COLUMNS),
        ]

        values: dict[str, pd.Series] = {
            "all_amounts_zero": (
                bills.eq(0).all(axis=1).astype("int64")
                * payments.eq(0).all(axis=1).astype("int64")
            ),
            "zero_payment_months": payments.eq(0).sum(axis=1),
            "delayed_months_count": repayment.gt(0).sum(axis=1),
            "payment_mean": payments.mean(axis=1),
            "payment_std": payments.std(axis=1, ddof=0),
            "bill_mean": bills.mean(axis=1),
            "bill_std": bills.std(axis=1, ddof=0),
            "payment_recent_minus_old": (
                payments.iloc[:, :3].mean(axis=1)
                - payments.iloc[:, 3:].mean(axis=1)
            ),
            "bill_recent_minus_old": (
                bills.iloc[:, :3].mean(axis=1)
                - bills.iloc[:, 3:].mean(axis=1)
            ),
            "max_delay": repayment.clip(lower=0).max(axis=1),
        }

        if set(self.features) & set(config.FE_REFINEMENT_RATIO_COLUMNS):
            values["bill_latest_to_limit"] = (
                bills.iloc[:, 0] / X["LIMIT_BAL"]
            )
            values["payment_mean_to_limit"] = (
                values["payment_mean"] / X["LIMIT_BAL"]
            )

        if "bill_change_to_limit" in self.features:
            values["bill_change_to_limit"] = (
                values["bill_recent_minus_old"] / X["LIMIT_BAL"]
            )
        
        derived = pd.DataFrame(
            {
                name: values[name]
                for name in self.features
            },
            index=X.index,
        )

        if not np.isfinite(
            derived.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                "Fitur turunan menghasilkan nilai non-finite."
            )

        return pd.concat(
            [X.copy(), derived],
            axis=1,
        )

    

    def get_feature_names_out(
        self,
        input_features: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Return original and selected derived feature names."""
        check_is_fitted(self, "feature_names_in_")

        if input_features is not None:
            if list(input_features) != self.feature_names_in_.tolist():
                raise ValueError(
                    "Nama fitur tidak sesuai input saat fit."
                )

        return np.concatenate(
            [
                self.feature_names_in_,
                np.asarray(self.features, dtype=object),
            ]
        )