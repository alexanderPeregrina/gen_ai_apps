    ## This script creates a vector data base using a list of text files and FAISS db, it uses Ollama embeddings
    # and retrieves the most relevant documents form the db base on an input query
    
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Directory loader
directory_loader = DirectoryLoader(path='./data', glob='**/*.txt', loader_cls= lambda path: TextLoader(path, encoding="utf-8"))
documents = directory_loader.load()
# Generate chunks from documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents) # {page_content : "text", metadata={source: "name.txt"}}
print(f"Total number of chunks: {len(chunks)}") 
# Generate embedding using Ollama
embeddings_func = OllamaEmbeddings(model="nomic-embed-text")

# Create vector data storage
faiss_retriever = FAISS.from_documents(documents=chunks, embedding=embeddings_func).as_retriever()

while True:
    query = input("Enter a query to search related documents or press quit to stop: ").strip()
    
    if query == 'quit':
        break
    
    results= faiss_retriever.invoke(input=query)
    for docu in results:
        print("###### Retrieved Document")
        print(docu.page_content)
