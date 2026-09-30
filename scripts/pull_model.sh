#!/usr/bin/env bash
# Pull the Llama 3.1 model into the running Ollama container/service.
set -euo pipefail
MODEL="${1:-llama3.1}"
if docker ps --format '{{.Names}}' | grep -q resume-ollama; then
    docker exec resume-ollama ollama pull "$MODEL"
else
    ollama pull "$MODEL"
fi
echo "Model '$MODEL' ready."
