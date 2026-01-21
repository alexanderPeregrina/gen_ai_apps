FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN mkdir -p /usr/local/ollama \
 && curl -fsSL https://ollama.com/download/Ollama-linux-amd64.tgz \
    | tar -xz -C /usr/local/ollama \
 && ln -s /usr/local/ollama/bin/ollama /usr/local/bin/ollama

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

# Copy entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose ports
EXPOSE 8501
EXPOSE 11434

CMD ["/app/entrypoint.sh"]
