ARG PYTHON_BASE_IMAGE=python:3.14-slim
FROM ${PYTHON_BASE_IMAGE}

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
# Scaffold empty package dirs so editable install succeeds with only
# pyproject.toml present. Editable mode writes an egg-link pointing at
# /app, so the real source copied below becomes live immediately —
# no re-install needed when only source changes.
RUN mkdir -p data pattern backtest visualization strategy pages signals db \
    && touch data/__init__.py pattern/__init__.py backtest/__init__.py \
             visualization/__init__.py strategy/__init__.py pages/__init__.py \
             signals/__init__.py db/__init__.py \
    && pip install --no-cache-dir -e ".[test]"

COPY . .

ENV PYTHONPATH=/app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
