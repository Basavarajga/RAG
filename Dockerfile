FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FINANCE_RAG_API_URL=http://127.0.0.1:8000/ask \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    API_PORT=8000 \
    STREAMLIT_PORT=8501

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .

# Create a non-root user
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
    
USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${API_PORT}/ >/dev/null || exit 1

CMD ["bash", "-c", "set -euo pipefail; uvicorn api:app --host 0.0.0.0 --port ${API_PORT} & api_pid=$!; streamlit run app.py --server.address=0.0.0.0 --server.port=${STREAMLIT_PORT} & ui_pid=$!; trap 'kill $api_pid $ui_pid 2>/dev/null || true' EXIT; wait -n $api_pid $ui_pid"]
