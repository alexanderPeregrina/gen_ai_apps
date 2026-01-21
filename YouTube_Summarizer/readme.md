# YouTube Summarizer App

This project provides a **YouTube Summarizer** that extracts transcripts or audio from YouTube videos and generates concise summaries using NLP techniques. It supports running locally with a Python virtual environment or inside a Docker container.

## Requirements
- Python 3.11
- `requirements.txt` provided in the repository
- FFmpeg (required for audio/video processing)
- (Optional) Docker with the included `Dockerfile`

---

## ⚙️ Installing FFmpeg

FFmpeg must be installed before running the app.  

### On Linux (Debian/Ubuntu)
```bash
sudo apt update
sudo apt install ffmpeg
ffmpeg -version
```

### On macOS (with Homebrew)
```
brew install ffmpeg
ffmpeg -version
```

### On Windows
- Download FFmpeg from: https://ffmpeg.org/download.html (ffmpeg.org in Bing)
- Extract the archive and add the bin folder to your PATH environment variable.
- Verify installation:

```
ffmpeg -version
```

## Running Locally (Virtual Environment)

### **Clone the repository**
   ```bash
   git clone -b feature/Optimize_PDF_RAG_system https://github.com/alexanderPeregrina/gen_ai_apps.git
   cd YouTube_Summarizer
   ```

### Create a virtual environment (Python 3.11)

```
python3.11 -m venv .venv
source .venv/bin/activate   # On Linux/Mac
.venv\Scripts\activate      # On Windows
```

### - Install dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```
```
streamlit run app/youtube_summarizer.py
```
## 🐳 Running with Docker
- Build the Docker image
```
docker build -t youtube-summarizer .
```
- Run the container
```
docker run --rm youtube-summarizer
```




