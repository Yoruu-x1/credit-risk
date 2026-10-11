FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# Library sistem untuk model berbasis OpenMP.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-runtime.txt ./

RUN python -m pip install --no-cache-dir -r requirements-runtime.txt \
    && python -m pip check

# Jalankan aplikasi sebagai pengguna biasa.
RUN useradd --create-home --uid 10001 appuser

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser app.py ./
COPY --chown=appuser:appuser models/blend_refinement_xgb75_v09/ ./models/blend_refinement_xgb75_v09/

USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]