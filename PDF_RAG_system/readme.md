# PDF RAG System

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline for working with PDF documents. It allows you to ingest, process, and query PDF files using modern NLP techniques.

## Requirements
- Python 3.11
- `requirements.txt` provided in the repository
- (Optional) Docker with the included `Dockerfile`

---

## Running Locally (Virtual Environment)

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/pdf-rag-system.git
   cd pdf-rag-system
   ```

2. **Create a virtual environment (Python 3.11)**
```bash
python3.11 -m venv .venv
source .venv/bin/activate   # On Linux/Mac
.venv\Scripts\activate      # On Windows
```
- Install dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```

- Run the script
```
streamlit run rag_pdf_system.py
```

## 🐳 Running with Docker
- Build the Docker image
```
docker build -t pdf-rag-system .
```
- Run the container
```
docker run --rm pdf-rag-system
```

- By default, the container runs with Python 3.11.

## ⚙️ Notes
- Ensure you have Python 3.11 installed locally if not using Docker.
- Use the virtual environment to avoid dependency conflicts.
- For Docker usage, you can extend the Dockerfile to include GPU support if needed.

