"""Pembacaan CSV, validasi schema, dan split sebelum preprocessing."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


@dataclass(frozen=True)
class DataSplit:
    """Menyimpan fitur dan target untuk kedua partisi."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def inspect_columns(path: Path = config.DATA_PATH) -> list[str]:
    """Membaca header CSV tanpa mengeksplorasi isi dataset."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")

    header = pd.read_csv(
        path,
        sep=config.CSV_SEPARATOR,
        encoding=config.CSV_ENCODING,
        nrows=0,
    )

    return header.columns.tolist()


def validate_schema(columns: list[str]) -> None:
    """Memastikan fitur dan target sesuai konfigurasi."""
    required = set(config.FEATURE_COLUMNS) | {
        config.RAW_TARGET_COLUMN
    }

    missing = required - set(columns)

    if missing:
        raise ValueError(
            f"Kolom wajib tidak ditemukan: {sorted(missing)}. "
            f"Kolom CSV: {columns}. "
            "Periksa nama target dan schema di config.py."
        )

    allowed = required | set(config.EXCLUDED_COLUMNS)
    unexpected = set(columns) - allowed

    if unexpected:
        raise ValueError(
            f"Kolom tambahan perlu ditinjau: {sorted(unexpected)}. "
            "Jangan memasukkannya sebagai fitur sebelum mengetahui maknanya."
        )


def validate_target(raw_target: pd.Series) -> pd.Series:
    """Memvalidasi target dengan 1=default dan 0=non-default."""
    if raw_target.isna().any():
        raise ValueError(
            "Target memiliki missing value. Target tidak boleh diimputasi."
        )

    try:
        target = pd.to_numeric(raw_target, errors="raise")
    except (ValueError, TypeError) as error:
        raise ValueError(
            "Target harus berupa label numerik 0 dan 1."
        ) from error

    if set(target.unique()) != {0, 1}:
        raise ValueError(
            "Target harus memiliki kedua kelas 0 dan 1. "
            f"Nilai ditemukan: {target.unique().tolist()}"
        )

    # Label dataset ini sudah memiliki arah yang benar.
    return target.astype("int64").rename(config.TARGET_NAME)


def load_and_split(path: Path = config.DATA_PATH) -> DataSplit:
    """Membagi fitur mentah sebelum imputasi, scaling, atau encoding."""
    columns = inspect_columns(path)
    validate_schema(columns)

    data = pd.read_csv(
        path,
        sep=config.CSV_SEPARATOR,
        encoding=config.CSV_ENCODING,
    )

    if data.empty:
        raise ValueError("Dataset kosong.")

    y = validate_target(data[config.RAW_TARGET_COLUMN])

    # Pemilihan kolom berdasarkan schema yang sudah ditetapkan.
    # Target, ID, dan indeks ekspor tidak masuk ke fitur.
    X = data.loc[:, list(config.FEATURE_COLUMNS)].copy()

    if y.value_counts().min() < 2:
        raise ValueError("Setiap kelas membutuhkan minimal dua baris.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    if set(y_train.unique()) != {0, 1} or set(y_test.unique()) != {0, 1}:
        raise ValueError("Setiap partisi harus memiliki kedua kelas.")

    return DataSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )