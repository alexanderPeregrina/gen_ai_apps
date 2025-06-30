from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from imageio_ffmpeg import get_ffmpeg_exe
import yt_dlp
import whisper
import streamlit as st
import nest_asyncio
import asyncio
# Patch event loop to avoid RuntimeError
nest_asyncio.apply()

import os

class LLMModels():
    """This class contains all required functionis and attributes to generate the RAG workflow pipeline. 
       It handles the split of documents, generates the vector database and sets up the QA chain
    """
    def __init__(self, llm_model_name = 'llama3.2', embedding_model_name = 'nomic-embed-text'):
        """ Init the LLM class object

        Args:
            llm_model_name (str, optional): Ollama model name to be used. Defaults to 'llama3.2'.
            embedding_model_name (str, optional): Ollama embedding model to be used. Defaults to 'nomic-embed-text'.
        """
        try:
            self.llm_model_name = llm_model_name
            self.embedding_model_name = embedding_model_name
            self.collection_name = self.embedding_model_name + "_chromaDB"
            self.llm = OllamaLLM(model=llm_model_name, temperature=0.1)
            self.embedding_model = OllamaEmbeddings(model=embedding_model_name)
            self.vector_store = None
            self.qa_chain = None
        except Exception as e:
            st.error(f"Error in model generation: {e}")
            
    def setup_knowledge_base(self, text, video_title):
        """Set up the knowloedge base from documents (chnuks) using a Chroma vector database

        Args:
            text (str): Text to be used to generate chunks and vector database.
            video_title (str): Title of if video to be used to name the vector database.

        Returns:
            List(Document): List of documents (chunks) obtanied from input text 
        """
        print("Setting up knowledge base ...")
        documents = self.__generate_documents(text, video_title)
        self.vector_store = self.__create_vector_store(documents)
        return documents
        
    def generate_summary(self, documents):
        """  Gnenerates summary based on documents using a summary chain """
        print("Generating summary ...")
        map_prompt = ChatPromptTemplate.from_template(
        """Write a concise summary of the following transcript {text}
           CONCISE SUMMARY:""")
        combine_prompt = ChatPromptTemplate.from_template(
        """Write a detailed summary of the following transcipt {text}
            Include: 
                - Main topics and key points
                - Important details and examples
                - Any conclusions or call to action
            DETAILED SUMMARY:
        """)
        summary_chain = load_summarize_chain(llm=self.llm, chain_type='map_reduce', verbose=True, map_prompt= map_prompt, combine_prompt= combine_prompt)
        return summary_chain.invoke(documents)
    
    def ask_model(self, question):
        """Creates a QA chain using langchain

        Args:
            question (str): Video related question that model will answer based upon the retrieved information
                            from vector database

        Returns:
            str: Result of qa chain 
        """
        if not self.qa_chain:
            self.qa_chain = self.__setup_qa_chain()
        print("Generating response ...")
        result = self.qa_chain.invoke({"question": question})
        return result
            
    # private functions
    def __generate_documents(self, text, video_title):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=[" ", "\n\n", "\n", ". ", ""])
        
        chunks = text_splitter.split_text(text=text)
        
        documents = [Document(page_content=chunk, metadata={"source": video_title}) for chunk in chunks]
        
        return documents
    
    def __create_vector_store(self, documents):
        return Chroma.from_documents(documents=documents, collection_name=self.collection_name, embedding=self.embedding_model)
    
    def __setup_qa_chain(self):
        if self.vector_store:
            memory_buffer = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            return ConversationalRetrievalChain.from_llm(llm=self.llm, retriever=self.vector_store.as_retriever(), memory=memory_buffer, verbose=True)
        else:
            st.error("Error: First setup a knowledge base")
            return None
    
class YoutubeSummarizer():
    def __init__(self, url):
        self.url = url
        self. ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'audio',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
            }],
        'ffmpeg_location': get_ffmpeg_exe() # Path to the folder containing ffmpeg.esxe
        }
        
    def process_video(self, llm_model):
        """Download audio, transcript and generate summary of self.url video

        Args:
            llm_model (str): Ollama LLM model name

        Returns:
            str: Summary of youtube video
        """
        file_name = self.__download_audio()
        
        print("Audio file path:", file_name)
        print("File exists:", os.path.exists(file_name))

        text = self.__transcribe_audio(file_name)
        documents = llm_model.setup_knowledge_base(text['text'], "the_doors")
        st.success("Vector Database succesfully created")
        summary = llm_model.generate_summary(documents)
        return summary

    def __download_audio(self):
        try:
            print("Downloading audio ...")
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(self.url, download=True)
                title = info.get('title', 'audio')
                filename = 'audio.mp3'
                st.success(f"{title} video completely downloaded")
            return filename

        except Exception as e:
            st.error(f"Error while downloading video: {str(e)}")
            raise e
            
    def __transcribe_audio(self, audio_path):
        try:
            print(f"Transcribing {audio_path}")
            model = whisper.load_model("base")
            result = model.transcribe(audio_path)
            os.remove(audio_path)
            st.success(f"Transcription completed!")
            return result
        except Exception as e:
            print(f"Error transcribing audio: {str(e)}")
            st.error(f"Error transcribing audio: {str(e)}")
            raise e
        


def main():
    st.title("🤖 YouTube Summarizer")

    if "llm_model" not in st.session_state:
        restart_llm_models()
    if "last_url" not in st.session_state:
        st.session_state.last_url = None

    # Display model info
    try:
        st.sidebar.info(
             f"LLM Models info:\n"
             f"- LLM model name: {st.session_state.llm_model.llm_model_name}\n"
             f"- Embedding model name: {st.session_state.llm_model.embedding_model_name}"
        )
    except Exception as e:
        st.error(f"Error initializing LLM model: {str(e)}")
        return

    url = st.text_input("🔍 Insert a YouTube video URL")
    if url:
        if st.button("Process Video") and st.session_state.last_url != url:
            restart_llm_models()
            st.session_state.last_url = url

            try:
                st.session_state.yt_summarizer = YoutubeSummarizer(url)
                with st.spinner("Processing video..."):
                    st.session_state.summary = st.session_state.yt_summarizer.process_video(st.session_state.llm_model)
                    st.success("Video processed successfully!")
                    last_url = url
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")

    # Display summary and Q&A
    if st.session_state.summary:
        st.markdown("### 📝 Summary:")
        st.write(st.session_state.summary["output_text"])
        question = st.text_input("Ask a related question:")
        st.session_state.question = question
        
    if st.session_state.question:
        if st.button("Generate response"): 
            with st.spinner("Generating response..."):
                try:
                    st.session_state.answer = st.session_state.llm_model.ask_model(st.session_state.question)                    
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
                    
    if st.session_state.answer:
        st.markdown("### 💬 Answer:")
        st.write(st.session_state.answer["answer"])
                    
def restart_llm_models():
    # Initialize session state
        st.session_state.llm_model = LLMModels()
        st.session_state.yt_summarizer = None
        st.session_state.summary = None
        st.session_state.question = None
        st.session_state.answer = None
                       
if __name__ == "__main__":
    main()