import inspect

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
import os
import PyPDF2
import uuid
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel
import vertexai
from dotenv import load_dotenv

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 50
DOCUMENTS_TO_RETRIEVE = 3


load_dotenv() 

class SimpleModelSelector:
    """Simple class to handle model selection"""

    embedding_models = {
        "chroma": {"name": "Chroma Default", "dimensions": 384, "model_name": None},
        "vertex": {
            "name": "Vertex AI Embeddings",
            "dimensions": 768,
            "model_name": "text-embedding-004",
        },
    }

    def __init__(self):
        self.llm_models = {"gemini-2.5-flash": "Gemini 2.5 Flash"}

    def select_models(self):
        st.sidebar.title("📚 Model Selection")

        llm = st.sidebar.radio(
            "Choose LLM Model:",
            options=list(self.llm_models.keys()),
            format_func=lambda x: self.llm_models[x],
        )

        embedding = st.sidebar.radio(
            "Choose Embedding Model:",
            options=list(self.embedding_models.keys()),
            format_func=lambda x: self.embedding_models[x]["name"],
        )

        return llm, embedding


class SimplePDFProcessor:
    def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def read_pdf(self, pdf_file):
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def create_chunks(self, text, pdf_file):
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if start > 0:
                start = start - self.chunk_overlap
            chunk = text[start:end]
            if end < len(text):
                last_period = chunk.rfind(".")
                if last_period != -1:
                    chunk = chunk[: last_period + 1]
                    end = start + last_period + 1
            chunks.append(
                {
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "metadata": {"source": pdf_file.name},
                }
            )
            start = end
        return chunks


class SimpleRAGSystem:
    def __init__(self, embedding_model="vertex", llm_model="gemini-2.5-flash"):
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.db = chromadb.Client(Settings(anonymized_telemetry=False))
        self.setup_embedding_function()
        self.collection = self.setup_collection()

    class VertexEmbeddingFunction:
        def __init__(self):
            vertexai.init(
                project=os.getenv("GCP_PROJECT"),
                location=os.getenv("GCP_LOCATION", "us-central1")
            )
            self.model = TextEmbeddingModel.from_pretrained(
                "text-embedding-004"
            )

    
        def __call__(self, input):
            resp = self.model.get_embeddings(input)
            return [r.values for r in resp]
        
    def setup_embedding_function(self):
        if self.embedding_model == "vertex":
            self.embedding_fn = self.VertexEmbeddingFunction()
        else:
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

 
    def setup_collection(self):
        collection_name = f"documents_{self.embedding_model}"
        collection = self.db.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"model": self.embedding_model},
        )
        st.info(f"Using or created collection for {self.embedding_model} embeddings")
        return collection

    def add_documents(self, chunks):
        try:
            if not self.collection:
                self.collection = self.setup_collection()
            self.collection.add(
                ids=[chunk["id"] for chunk in chunks],
                documents=[chunk["text"] for chunk in chunks],
                metadatas=[chunk["metadata"] for chunk in chunks],
            )
            return True
        except Exception as e:
            st.error(f"Error adding documents: {str(e)}")
            return False

    def query_documents(self, query, pdf_filename, n_results=DOCUMENTS_TO_RETRIEVE):
        try:
            if not self.collection:
                raise ValueError("No collection available")
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where={"source": pdf_filename}
            )
            return results
        except Exception as e:
            st.error(f"Error querying documents: {str(e)}")
            return None

    def generate_response(self, query, context):
        try:
            prompt = f"""
            Based on the following context, please answer the question.
            If you can't find the answer in the context, say so, or I don't know.

            Context: {context}

            Question: {query}

            Answer:
            """.strip()

            vertexai.init(project=os.getenv("GCP_PROJECT"), location=os.getenv("GCP_LOCATION", "us-central1"))
            model = GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return None

    def get_embedding_info(self):
        model_info = SimpleModelSelector.embedding_models[self.embedding_model]
        return {
            "name": model_info["name"],
            "dimensions": model_info["dimensions"],
            "model": self.embedding_model
        }


def main():
    st.title("🤖 Simple PDF RAG System (Vertex AI)")

    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "current_embedding_model" not in st.session_state:
        st.session_state.current_embedding_model = None
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None

    model_selector = SimpleModelSelector()
    llm_model, embedding_model = model_selector.select_models()

    if embedding_model != st.session_state.current_embedding_model:
        st.session_state.processed_files.clear()
        st.session_state.current_embedding_model = embedding_model
        st.session_state.rag_system = None
        st.warning("Embedding model changed. Please re-upload your documents.")

    try:
        if st.session_state.rag_system is None:
            st.session_state.rag_system = SimpleRAGSystem(embedding_model, llm_model)
        embedding_info = st.session_state.rag_system.get_embedding_info()
        st.sidebar.info(
            f"Current Embedding Model:\n"
            f"- Name: {embedding_info['name']}\n"
            f"- Dimensions: {embedding_info['dimensions']}"
        )
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return

    pdf_file = st.file_uploader("Upload PDF", type="pdf")

    if pdf_file and pdf_file.name not in st.session_state.processed_files:
        st.session_state.current_pdf = pdf_file.name
        processor = SimplePDFProcessor()
        with st.spinner("Processing PDF..."):
            try:
                text = processor.read_pdf(pdf_file)
                chunks = processor.create_chunks(text, pdf_file)
                if st.session_state.rag_system.add_documents(chunks):
                    st.session_state.processed_files.add(pdf_file.name)
                    st.success(f"Successfully processed {pdf_file.name}")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")

    if st.session_state.processed_files:
        st.markdown("---")
        st.subheader("🔍 Query Your Documents")
        query = st.text_input("Ask a question:")

        if query:
            with st.spinner("Generating response..."):
                pdf_name = st.session_state.get("current_pdf")
                if not pdf_name:
                    st.warning("Please upload a PDF first.")
                else:
                    results = st.session_state.rag_system.query_documents(query, pdf_name)
                    if results and results["documents"]:
                        response = st.session_state.rag_system.generate_response(
                            query, results["documents"][0]
                        )
                        if response:
                            st.markdown("### 📝 Answer:")
                            st.write(response)
                            with st.expander("View Source Passages"):
                                for idx, doc in enumerate(results["documents"][0], 1):
                                    st.markdown(f"**Passage {idx}:**")
                                    st.info(doc)
    else:
        st.info("👆 Please upload a PDF document to get started!")

if __name__ == "__main__":
    main()
