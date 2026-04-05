import streamlit as st
import chromadb
from chromadb.config import Settings
import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel
import yt_dlp
import os
from imageio_ffmpeg import get_ffmpeg_exe
import uuid
from google.cloud import speech, storage
from dotenv import load_dotenv

load_dotenv()

# -------------------------
# CONFIG
# -------------------------
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 100
TOP_K = 3

# -------------------------
# EMBEDDING FUNCTION
# -------------------------
class VertexEmbeddingFunction:
    def __init__(self):
        vertexai.init(
            project=os.getenv("GCP_PROJECT"),
            location=os.getenv("GCP_LOCATION", "us-central1")
        )
        self.model = TextEmbeddingModel.from_pretrained("text-embedding-004")

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        resp = self.model.get_embeddings(input)
        return [r.values for r in resp]


# -------------------------
# TEXT PROCESSOR
# -------------------------
class TextProcessor:
    def create_chunks(self, text):
        chunks = []
        start = 0

        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end]

            chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk
            })

            start = end - CHUNK_OVERLAP

        return chunks


# -------------------------
# RAG SYSTEM
# -------------------------
class VertexRAG:
    def __init__(self):
        self.embedding_fn = VertexEmbeddingFunction()

        self.client = chromadb.Client(
            Settings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_or_create_collection(
            name="youtube_rag",
            embedding_function=self.embedding_fn
        )

        vertexai.init(
            project=os.getenv("GCP_PROJECT"),
            location=os.getenv("GCP_LOCATION", "us-central1")
        )

        self.llm = GenerativeModel("gemini-2.5-flash")

    def add_documents(self, chunks, source):
        self.collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[{"source": source} for _ in chunks]
        )

    def query(self, question, source):
        results = self.collection.query(
            query_texts=[question],
            n_results=TOP_K,
            where={"source": source}
        )
        return results["documents"][0]

    def generate_answer(self, question, context):
        prompt = f"""
        Answer the question based ONLY on the context.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """

        response = self.llm.generate_content(prompt)
        return response.text

    def summarize(self, text):
        prompt = f"""
        Summarize the following transcript clearly:

        {text}
        """
        response = self.llm.generate_content(prompt)
        return response.text


# -------------------------
# YOUTUBE PROCESSOR
# -------------------------
class YoutubeProcessor:
    def __init__(self, url):
        self.url = url
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': 'audio',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3'
            }],
            'ffmpeg_location': get_ffmpeg_exe()
        }

    def download_audio(self):
        with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
            info = ydl.extract_info(self.url, download=True)
            title = info.get("title", "audio")
            filename = ydl.prepare_filename(info)
            filename = os.path.splitext(filename)[0] + ".mp3"
        return filename, title
    
    def upload_to_gcs(self, file_path, bucket_name):
        storage_client = storage.Client(project=os.getenv("GCP_PROJECT"))
        bucket = storage_client.bucket(bucket_name)
        blob_name = f"audio/{uuid.uuid4()}.mp3"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(file_path)
        return f"gs://{bucket_name}/{blob_name}", blob_name
    
    def transcribe(self, audio_path):
        bucket_name = os.getenv("GCS_BUCKET")

        # 1. Upload to GCS
        gcs_uri, blob_name = self.upload_to_gcs(audio_path, bucket_name)

        client = speech.SpeechClient()

        audio = speech.RecognitionAudio(uri=gcs_uri)

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.MP3,
            language_code="en-US",
            alternative_language_codes=["es-MX"],
            enable_automatic_punctuation=True,
            model="latest_long",
        )

        # 2. Async transcription
        operation = client.long_running_recognize(
            config=config,
            audio=audio
        )

        response = operation.result(timeout=1200)

        # 3. Build transcript
        text = " ".join(
            result.alternatives[0].transcript
            for result in response.results
        )

        # 4. Cleanup local file
        os.remove(audio_path)

        # 5. Cleanup GCS file (optional but recommended)
        storage_client = storage.Client(project=os.getenv("GCP_PROJECT"))
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.delete()

        return {"text": text}


# -------------------------
# STREAMLIT APP
# -------------------------
def main():
    st.title("🎥 YouTube RAG Summarizer (Vertex AI)")

    if "rag" not in st.session_state:
        st.session_state.rag = VertexRAG()
        st.session_state.current_video = None
        st.session_state.summary = None

    url = st.text_input("Enter YouTube URL")

    if st.button("Process Video") and url:
        processor = YoutubeProcessor(url)

        with st.spinner("Downloading audio..."):
            file_path, title = processor.download_audio()

        with st.spinner("Transcribing..."):
            text = processor.transcribe(file_path)

        with st.spinner("Creating embeddings..."):
            tp = TextProcessor()
            chunks = tp.create_chunks(text['text'])
            st.session_state.rag.add_documents(chunks, title)
            st.session_state.current_video = title

        with st.spinner("Generating summary..."):
            summary = st.session_state.rag.summarize(text)
            st.session_state.summary = summary

    # -------------------------
    # SHOW SUMMARY
    # -------------------------
    if st.session_state.summary:
        st.markdown("### 📝 Summary")
        st.write(st.session_state.summary)

        question = st.text_input("Ask a question")

        if question:
            with st.spinner("Thinking..."):
                docs = st.session_state.rag.query(
                    question,
                    st.session_state.current_video
                )

                answer = st.session_state.rag.generate_answer(
                    question,
                    docs
                )

                st.markdown("### 💬 Answer")
                st.write(answer)


if __name__ == "__main__":
    main()