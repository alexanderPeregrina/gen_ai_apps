from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Directory loader
directory_loader = DirectoryLoader(path='./data', glob='**/*.txt', loader_cls= lambda path: TextLoader(path, encoding="utf-8"))
documents = directory_loader.load()
# Generate chunks from documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents) # {page_content : "text", metadata={source: "name.txt"}}
print(f"Total number of chunks: {len(chunks)}") 
# Generate embedding using Ollama
embeddings_func = OllamaEmbeddings(model="nomic-embed-text")

ollama_model = OllamaLLM(model='llama3.2')

# Create vector data storage
faiss_retriever = FAISS.from_documents(documents=chunks, embedding=embeddings_func).as_retriever()


prompt_template = PromptTemplate( input_variables=["context", "question"],
    template="""
    You are a helpful assistant. Use the following context to answer the question.

    Context:
    {context}

    Question:
    {question}

    Answer:"""
    )

# Chain the prompt, model, and output parser
 # Create the chain
chain = (
        {"context": faiss_retriever, "question": RunnablePassthrough()}
        | prompt_template
        | ollama_model
        | StrOutputParser()
    )


while True:
    query = input("Enter a query to search related documents or press quit to stop: ").strip()
    
    if query == 'quit':
        break
    
    response = chain.invoke(query)
    print(response)