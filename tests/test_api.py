"""Pengujian kontrak API dan konsistensi artefak."""

from collections.abc import Iterator
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app
from src import config


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """Jalankan startup dan shutdown API selama pengujian."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def payload() -> dict:
    """Gunakan contoh development yang tersimpan dalam schema."""
    document = json.loads(
        config.FINAL_SCHEMA_PATH.read_text(encoding="utf-8")
    )
    return dict(document["example_request"])


def test_prediction_matches_model(
    client: TestClient,
    payload: dict,
) -> None:
    """API harus menghasilkan skor dan keputusan artefak."""
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    body = response.json()
    frame = pd.DataFrame(
        [payload],
        columns=list(config.FEATURE_COLUMNS),
    )
    bundle = app.state.bundle
    expected = float(
        bundle["model"].predict_proba(frame)[0, 1]
    )

    assert body["probability_default"] == pytest.approx(
        expected, abs=1e-10
    )
    assert body["threshold"] == bundle["threshold"]
    assert body["predicted_default"] == (
        expected >= bundle["threshold"]
    )


@pytest.mark.parametrize(
    "invalid_fields",
    [
        {"SEX": 999},
        {"LIMIT_BAL": 0},
        {"AGE": True},
        {"ID": 123},
    ],
)
def test_invalid_input_is_rejected(
    client: TestClient,
    payload: dict,
    invalid_fields: dict,
) -> None:
    """Tolak kategori, nilai, tipe, dan fitur yang tidak didukung."""
    response = client.post(
        "/predict",
        json={**payload, **invalid_fields},
    )
    assert response.status_code == 422


def test_explanation_reconstructs_prediction(
    client: TestClient,
    payload: dict,
) -> None:
    """Base, kontribusi terpilih, dan residual harus cocok dengan skor."""
    response = client.post(
        "/predict?explain=true",
        json=payload,
    )
    assert response.status_code == 200

    body = response.json()
    explanation = body["explanation"]

    reconstructed = (
        explanation["base_probability"]
        + (
            sum(
                reason["contribution_pp"]
                for reason in explanation["reasons"]
            )
            + explanation["remaining_contribution_pp"]
        ) / 100
    )

    assert reconstructed == pytest.approx(
        body["probability_default"],
        abs=config.SHAP_ADDITIVITY_ATOL,
    )
    assert explanation["reconstruction_error"] <= (
        config.SHAP_ADDITIVITY_ATOL
    )


def test_categorical_enum_is_documented(
    client: TestClient,
) -> None:
    """Kategori harus muncul sebagai enum dalam OpenAPI."""
    schema = client.get("/openapi.json").json()
    assert schema["components"]["schemas"]["SEXCode"]["enum"] == [1, 2]