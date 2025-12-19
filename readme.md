# 🧠 Generative AI and Large Language Model Applications

This repository showcases a collection of Generative AI and Large Language Model (LLM) applications built in Python using open-source models and frameworks.

## Projects Included
* Chatbots using open-source models via the Hugging Face transformers library.
*  Chatbot powered by Ollama's open-source models.
* Vector database creation using ChromaDB and embeddings.
* RAG system for news retrieval using FAISS vector database.
* RAG system for question-answering over PDF documents.
* YouTube summarizer and Q&A using LangChain.
* LoRA fine-tuning on open-source models for ML classification tasks.
* Code review agent (in progress).

## Getting Started

**Note:** All scripts were tested using Python 11, make sure to use this version of python when creating environment. To run any script in this repository, follow these steps:

### 1. Create a Virtual Environment

```
python -m venv <v_env_name>
```

### 2. Activate the Environment
* On **Windows**:

```
.\<venv_name>\Scripts\activate

```
* On **macOS/Linux**:
```
source <venv_name>/bin/activate

```

## 3. Install Dependencies

```
pip install -r requirements.txt
```

## Install Ollama

Install Ollama from its official web site:  
https://ollama.com/download

This repository uses the following Ollama models, install them using the following comands (make sure ollama application is already installed in your PC):

```
ollama run llama3.2
```

```
ollama run deepseek-r1
```

