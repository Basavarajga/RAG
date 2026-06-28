FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_PORT=8501

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the FAISS index from the checked-in corpus so the image is ready to serve.
RUN python src/embeddings.py \
    && useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
    
USER appuser

CMD ["bash", "-c", "set -euo pipefail; API_PORT=${PORT:-8000}; export FINANCE_RAG_API_URL=http://127.0.0.1:${API_PORT}/ask; uvicorn api:app --host 0.0.0.0 --port ${API_PORT} & api_pid=$!; streamlit run app.py --server.address=0.0.0.0 --server.port=${STREAMLIT_PORT} & ui_pid=$!; trap 'kill $api_pid $ui_pid 2>/dev/null || true' EXIT; wait -n $api_pid $ui_pid"]
