"""API prediksi default dan penjelasan model final."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from enum import IntEnum
from importlib.metadata import version
import json
from threading import Lock
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    create_model,
    model_validator,
)

from src import config
from src.explain import (
    CreditExplainer,
    file_sha256,
    load_final_bundle,
    reason_table,
)

schema_document = json.loads(
    config.FINAL_SCHEMA_PATH.read_text(encoding="utf-8")
)


class RequestBase(BaseModel):
    """Tolak fitur tambahan, nilai non-finite, dan boolean."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        json_schema_extra={
            "example": schema_document["example_request"],
        },
    )

    @model_validator(mode="before")
    @classmethod
    def reject_boolean_features(cls, data: Any) -> Any:
        """Boolean bukan nilai fitur numerik dalam kontrak ini."""
        if isinstance(data, dict) and any(
            isinstance(value, bool) for value in data.values()
        ):
            raise ValueError("Nilai fitur tidak boleh boolean.")
        return data


def build_request_model() -> type[BaseModel]:
    """Bangun schema dari fitur mentah dan kategori encoder final."""
    fields = {}
    codes = schema_document["categorical_codes"]

    for feature in config.FEATURE_COLUMNS:
        if feature in codes:
            enum_type = IntEnum(
                f"{feature}Code",
                {
                    (
                        f"CODE_{code}"
                        if code >= 0
                        else f"CODE_NEG_{abs(code)}"
                    ): code
                    for code in codes[feature]
                },
            )
            fields[feature] = (enum_type, Field(...))

        elif feature == "AGE":
            fields[feature] = (
                int,
                Field(..., gt=0, strict=True),
            )

        elif feature == "LIMIT_BAL":
            fields[feature] = (
                float,
                Field(..., gt=0),
            )

        elif feature in config.PAYMENT_COLUMNS:
            fields[feature] = (
                float,
                Field(..., ge=0),
            )

        else:
            # Tagihan negatif dipertahankan karena ada
            # dalam domain data; tidak dipotong menjadi nol.
            fields[feature] = (float, Field(...))

    return create_model(
        "CreditRequest",
        __base__=RequestBase,
        **fields,
    )


CreditRequest = build_request_model()


class Reason(BaseModel):
    """Satu kontribusi fitur yang dapat dibaca."""

    feature: str
    value: float
    value_label: str
    contribution_pp: float
    definition: str


class ExplanationResponse(BaseModel):
    """Kontribusi terbesar beserta residual agar jumlah tetap lengkap."""

    base_probability: float
    reasons: list[Reason]
    remaining_contribution_pp: float
    reconstruction_error: float


class PredictionResponse(BaseModel):
    """Skor dan flag berdasarkan threshold terkunci."""

    model_version: str
    probability_default: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    predicted_default: bool
    risk_flag: Literal["high_risk", "lower_risk"]
    explanation: ExplanationResponse | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Muat dan verifikasi model serta background sekali saat startup."""
    bundle, metadata = load_final_bundle()

    explanation_metadata = json.loads(
        config.SHAP_METADATA_PATH.read_text(encoding="utf-8")
    )

    checks = (
        schema_document["model_sha256"] == metadata["model_sha256"],
        explanation_metadata["model_sha256"] == metadata["model_sha256"],
        explanation_metadata["background_sha256"]
        == file_sha256(config.SHAP_BACKGROUND_PATH),
        explanation_metadata["schema_sha256"]
        == file_sha256(config.FINAL_SCHEMA_PATH),
        explanation_metadata["additivity_atol"]
        == config.SHAP_ADDITIVITY_ATOL,
    )
    if not all(checks):
        raise RuntimeError("Artefak penjelasan tidak cocok dengan model.")

    for package, expected in explanation_metadata["versions"].items():
        if version(package) != expected:
            raise RuntimeError(f"Versi penjelasan berubah: {package}")

    background = pd.read_csv(config.SHAP_BACKGROUND_PATH)
    explainer = CreditExplainer(bundle["model"], background)

    # Smoke check saat startup; bukan evaluasi test.
    explainer.explain(background.iloc[:1])

    app.state.bundle = bundle
    app.state.explainer = explainer
    app.state.inference_lock = Lock()

    yield


app = FastAPI(
    title="Credit Default Prediction API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Laporkan kesiapan model setelah startup berhasil."""
    return {
        "status": "ready",
        "model_version": app.state.bundle["candidate"],
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    response_model_exclude_none=True,
)
def predict(
    request: CreditRequest,
    explain: bool = False,
) -> PredictionResponse:
    """Prediksi satu nasabah dengan penjelasan opsional."""
    frame = pd.DataFrame(
        [request.model_dump(mode="json")],
        columns=list(config.FEATURE_COLUMNS),
    )

    bundle = app.state.bundle
    explanation_response = None

    # Serialisasi akses untuk implementasi CPU sederhana ini.
    with app.state.inference_lock:
        if explain:
            result = app.state.explainer.explain(frame)
            probability = float(result.probabilities[0])
            table = reason_table(result)

            top_reasons = table.head(config.API_TOP_REASONS)
            remaining = float(
                table.iloc[config.API_TOP_REASONS:][
                    "contribution_pp"
                ].sum()
            )

            reconstructed = (
                result.base_probability
                + table["contribution_pp"].sum() / 100
            )

            explanation_response = ExplanationResponse(
                base_probability=result.base_probability,
                reasons=[
                    Reason(**row)
                    for row in top_reasons.to_dict(orient="records")
                ],
                remaining_contribution_pp=remaining,
                reconstruction_error=float(
                    abs(reconstructed - probability)
                ),
            )
        else:
            probability = float(
                bundle["model"].predict_proba(frame)[0, 1]
            )

    predicted_default = probability >= bundle["threshold"]

    return PredictionResponse(
        model_version=bundle["candidate"],
        probability_default=probability,
        threshold=bundle["threshold"],
        predicted_default=predicted_default,
        risk_flag=(
            "high_risk" if predicted_default else "lower_risk"
        ),
        explanation=explanation_response,
    )