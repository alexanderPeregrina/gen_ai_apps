#!/bin/bash
set -e

# Start Ollama server in background
/usr/local/bin/ollama serve &

sleep 5

# Trigger signin flow (prints URL to logs)
echo "🔑 Ollama Cloud login required. Please open the URL below to link this container:"
/usr/local/bin/ollama signin

# Preload models after signin
/usr/local/bin/ollama pull nomic-embed-text
/usr/local/bin/ollama pull gpt-oss:120b-cloud
/usr/local/bin/ollama pull qwen3-vl:235b-cloud

# Start Streamlit app
exec streamlit run rag_pdf_system.py --server.port=8501 --server.address=0.0.0.0
