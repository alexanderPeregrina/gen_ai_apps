
# Dockerfile
FROM python:3.11.8

# Install system dependencies

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    ca-certificates \
    lsb-release \
    sudo \
    && apt-get install -y ffmpeg \
    && rm -rf /var/lib/apt/lists/*


# Install Ollama and install llama3.2 model
RUN curl -fsSL https://ollama.com/install.sh | bash

RUN ollama serve & sleep 5 && ollama run llama3.2 && ollama run nomic-embed-text || true

# Set working directory
WORKDIR /app

# Copy app files
COPY app/ /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose ports
EXPOSE 8501 11434

# Start both Ollama and Streamlit
CMD sh -c "ollama serve & streamlit run youtube_summarizer.py --server.port=8501 --server.address=0.0.0.0"


