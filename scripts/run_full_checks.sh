#!/usr/bin/env bash
set -euo pipefail

QUESTION="Is organic milk available and what is the return policy?"

python -m compileall -q .
python src/build_corpus.py
python src/embeddings.py
python src/rag_pipeline.py -q "What is the return policy?"
python evaluation/evaluate.py

uvicorn api:app --host 127.0.0.1 --port 8001 >/tmp/uvicorn_retail_ai.log 2>&1 &
UVICORN_PID=$!

cleanup() {
  kill "$UVICORN_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

READY=0
for _ in $(seq 1 30); do
  if curl -sSf http://127.0.0.1:8001/ >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 1
done

if [ "$READY" -eq 0 ]; then
  echo "[ERROR] uvicorn did not become ready within 30 seconds." >&2
  echo "[ERROR] Check logs at /tmp/uvicorn_retail_ai.log" >&2
  exit 1
fi

curl -sSf -X POST http://127.0.0.1:8001/ask \
  -H 'Content-Type: application/json' \
  -d "{\"question\":\"${QUESTION}\"}" >/tmp/api_ask_post.json

echo "[INFO] API POST /ask response:"
cat /tmp/api_ask_post.json
echo
