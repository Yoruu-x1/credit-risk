"""Penjelasan probabilitas ensemble dengan TreeSHAP per anggota."""

from dataclasses import dataclass
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import VotingClassifier

from src import config
from src.preprocess import get_feature_mapping


def file_sha256(path: Path) -> str:
    """Hitung hash file secara bertahap."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_final_bundle() -> tuple[dict[str, Any], dict[str, Any]]:
    """Muat model terkunci dan periksa kompatibilitas inference."""
    metadata = json.loads(
        config.FINAL_METADATA_PATH.read_text(encoding="utf-8")
    )

    if not metadata["test_evaluated"]:
        raise RuntimeError("Evaluasi final belum selesai.")

    if file_sha256(config.FINAL_MODEL_PATH) != metadata["model_sha256"]:
        raise RuntimeError("Hash model tidak cocok.")

    for package, expected in metadata["versions"].items():
        current = version(package)
        if current != expected:
            raise RuntimeError(
                f"Versi {package} berubah: {expected} -> {current}"
            )

    # config mendapat tambahan pengaturan deployment.
    # Implementasi transformasi model harus tetap sama.
    for filename in ("feature_engineering.py", "preprocess.py"):
        path = config.PROJECT_ROOT / "src" / filename
        if file_sha256(path) != metadata["source_sha256"][filename]:
            raise RuntimeError(f"Transformasi model berubah: {filename}")

    bundle = joblib.load(config.FINAL_MODEL_PATH)
    model = bundle["model"]

    compatible = (
        bundle["candidate"] == metadata["candidate"]
        == config.FINAL_MODEL_NAME
        and bundle["threshold"] == metadata["threshold"]
        and bundle["positive_class"] == 1
        and model.classes_.tolist() == [0, 1]
        and model.feature_names_in_.tolist()
        == list(config.FEATURE_COLUMNS)
    )
    if not compatible:
        raise RuntimeError("Kontrak artefak model tidak cocok.")

    return bundle, metadata


@dataclass
class ExplanationBatch:
    """Probabilitas dan kontribusi terkelompok untuk satu batch."""

    probabilities: np.ndarray
    base_probability: float
    contributions: pd.DataFrame
    feature_values: pd.DataFrame
    checks: pd.DataFrame


class CreditExplainer:
    """Gabungkan penjelasan input model sesuai bobot ensemble."""

    def __init__(
        self,
        model: VotingClassifier,
        background: pd.DataFrame,
    ) -> None:
        self.model = model
        background = self._validate_frame(background)

        names = list(model.named_estimators_)
        weights = np.asarray(model.weights, dtype=float)

        if len(weights) != len(names) or not np.isfinite(weights).all():
            raise ValueError("Bobot ensemble tidak valid.")
        if (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("Bobot harus nonnegatif dengan jumlah positif.")

        self.weights = dict(
            zip(names, weights / weights.sum(), strict=True)
        )
        self.explainers = {}
        self.mappings = {}

        for name, pipeline in model.named_estimators_.items():
            engineered = pipeline.named_steps[
                "feature_engineering"
            ].transform(background)
            encoded = pipeline.named_steps["preprocess"].transform(
                engineered
            )
            mapping = get_feature_mapping(
                pipeline.named_steps["preprocess"]
            )

            if encoded.columns.tolist() != mapping[
                "encoded_feature"
            ].tolist():
                raise RuntimeError("Urutan mapping berbeda dari input model.")

            self.mappings[name] = mapping
            self.explainers[name] = shap.TreeExplainer(
                pipeline.named_steps["model"],
                data=encoded.to_numpy(dtype=float),
                model_output="probability",
                feature_perturbation="interventional",
            )

    @staticmethod
    def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
        """Validasi schema mentah dan nilai finite."""
        if frame.empty or frame.columns.has_duplicates:
            raise ValueError("Input kosong atau nama kolom duplikat.")

        if set(frame.columns) != set(config.FEATURE_COLUMNS):
            raise ValueError("Input harus berisi tepat seluruh fitur mentah.")

        frame = frame.loc[:, list(config.FEATURE_COLUMNS)].copy()

        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError("Input mengandung nilai non-finite.")

        return frame

    def explain(self, frame: pd.DataFrame) -> ExplanationBatch:
        """Hitung kontribusi dan periksa rekonstruksi probabilitas."""
        frame = self._validate_frame(frame)
        total = pd.DataFrame(index=frame.index)
        feature_values = frame.copy()
        base_probability = 0.0
        checks = []

        for name, pipeline in self.model.named_estimators_.items():
            engineered = pipeline.named_steps[
                "feature_engineering"
            ].transform(frame)

            for column in engineered.columns:
                if column not in feature_values:
                    feature_values[column] = engineered[column]

            encoded = pipeline.named_steps["preprocess"].transform(
                engineered
            )
            explainer = self.explainers[name]

            values = np.asarray(
                explainer.shap_values(
                    encoded.to_numpy(dtype=float),
                    check_additivity=True,
                ),
                dtype=float,
            )
            expected = np.asarray(
                explainer.expected_value, dtype=float
            ).reshape(-1)

            # Beberapa model mengembalikan satu output;
            # model lain mengembalikan kedua kelas.
            if (
                values.ndim == 3
                and values.shape[2] == 2
                and expected.size == 2
            ):
                values = values[:, :, 1]
                base = float(expected[1])
            elif values.ndim == 2 and expected.size == 1:
                base = float(expected[0])
            else:
                raise RuntimeError(
                    f"Bentuk output SHAP tidak didukung: {values.shape}"
                )

            if values.shape != encoded.shape:
                raise RuntimeError("Dimensi SHAP berbeda dari input model.")

            probabilities = pipeline.predict_proba(frame)[:, 1]
            error = float(np.max(np.abs(
                base + values.sum(axis=1) - probabilities
            )))

            if (
                not np.isfinite(error)
                or error > config.SHAP_ADDITIVITY_ATOL
            ):
                raise RuntimeError(
                    f"Rekonstruksi SHAP {name} gagal: {error}"
                )

            checks.append({
                "model": name,
                "max_absolute_error": error,
            })

            # Jumlahkan semua kontribusi OHE, termasuk kolom
            # kategori yang bernilai nol untuk observasi tersebut.
            grouped = pd.DataFrame(
                values,
                index=frame.index,
                columns=self.mappings[name][
                    "original_feature"
                ].tolist(),
            ).T.groupby(level=0, sort=False).sum().T

            total = total.add(
                grouped * self.weights[name],
                fill_value=0.0,
            )
            base_probability += self.weights[name] * base

        probabilities = self.model.predict_proba(frame)[:, 1]
        error = float(np.max(np.abs(
            base_probability
            + total.sum(axis=1).to_numpy()
            - probabilities
        )))

        if (
            not np.isfinite(error)
            or error > config.SHAP_ADDITIVITY_ATOL
        ):
            raise RuntimeError(
                f"Rekonstruksi SHAP ensemble gagal: {error}"
            )

        checks.append({
            "model": "weighted_blend",
            "max_absolute_error": error,
        })

        return ExplanationBatch(
            probabilities=probabilities,
            base_probability=base_probability,
            contributions=total,
            feature_values=feature_values,
            checks=pd.DataFrame(checks),
        )


def reason_table(
    result: ExplanationBatch,
    row_position: int = 0,
) -> pd.DataFrame:
    """Tampilkan nilai aktual dan kontribusi dalam percentage points."""
    rows = []

    for feature, contribution in result.contributions.iloc[
        row_position
    ].items():
        value = float(
            result.feature_values.iloc[row_position][feature]
        )
        label = f"{value:.6g}"

        if feature in config.CATEGORICAL_COLUMNS:
            code = int(value)
            label = config.CATEGORY_LABELS.get(
                feature, {}
            ).get(code, f"Code {code}")

            if feature in config.REPAYMENT_STATUS_COLUMNS:
                if code > 0:
                    label = f"Keterlambatan {code} bulan"
                elif code == -1:
                    label = "Pembayaran tepat waktu (kode -1)"
                else:
                    label = f"Status repayment kode {code}"

        rows.append({
            "feature": feature,
            "value": value,
            "value_label": label,
            "contribution_pp": float(contribution * 100),
            "definition": config.ENGINEERED_FEATURE_DEFINITIONS.get(
                feature, feature
            ),
        })

    table = pd.DataFrame(rows)
    order = table["contribution_pp"].abs().sort_values(
        ascending=False
    ).index

    return table.loc[order].reset_index(drop=True)