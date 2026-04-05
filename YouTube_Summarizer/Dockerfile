# -------------------------
# Base Image
# -------------------------
FROM python:3.11-slim

# -------------------------
# System dependencies
# -------------------------
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Set working directory
# -------------------------
WORKDIR /app

# -------------------------
# Copy requirements first (for caching)
# -------------------------
COPY requirements.txt .

# -------------------------
# Install Python dependencies
# -------------------------
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -------------------------
# Copy app code
# -------------------------
COPY . .

# -------------------------
# Streamlit config (optional but recommended)
# -------------------------
RUN mkdir -p /root/.streamlit && \
    echo "\
[server]\n\
headless = true\n\
port = 8501\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml

# -------------------------
# Expose port
# -------------------------
EXPOSE 8501

# -------------------------
# Run app
# -------------------------
CMD ["streamlit", "run", "app/youtube_summarizer.py", "--server.port=8501", "--server.address=0.0.0.0"]