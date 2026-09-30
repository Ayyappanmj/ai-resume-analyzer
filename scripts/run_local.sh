#!/usr/bin/env bash
# Run backend + frontend locally without Docker (assumes Python 3.11 + Ollama installed).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 -m venv "$ROOT/.venv" 2>/dev/null || true
source "$ROOT/.venv/bin/activate"
pip install -q -r "$ROOT/backend/requirements.txt" -r "$ROOT/frontend/requirements.txt"
python -m spacy download en_core_web_sm

export OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
export DATABASE_URL="${DATABASE_URL:-}"

(cd "$ROOT/backend" && uvicorn app.main:app --reload --port 8000) &
BACK_PID=$!
trap "kill $BACK_PID" EXIT

sleep 2
(cd "$ROOT/frontend" && API_URL=http://localhost:8000 streamlit run app.py)
