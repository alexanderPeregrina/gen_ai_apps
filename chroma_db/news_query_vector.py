import chromadb
import ollama
from chromadb.utils import embedding_functions
import os

def get_document_list(directory_path):
    """ Get a list with information about the documents contained in the given directory

    Args:
        directory_path (str): path of the directory where .txt file are contained

    Returns:
        list(dir): List of dictionaries containing id, text
    """
    print("========= LOADING DOCUMENTS FROM PATH =============")
    documents = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.txt'):
            with open(os.path.join(directory_path, file_name), mode='r', encoding='utf8') as news_file:
                documents.append({'id': file_name,
                                  'text': news_file.read()})
    return documents

# function to split the texts into chunks of size {chunk_size}
# this will return a list of texts
def split_text(text, chunk_size=1000, chunk_overlap=20):
    text_lenght = len(text)
    if text_lenght < chunk_size:
        return [text]
    else:
        start = 0
        chunks = []
        while start < text_lenght:
            end = min(text_lenght, start + chunk_size)
            chunks.append(text[start : end])
            start += chunk_size
            start -= chunk_overlap
        return chunks
    
def query_documents(question, n_results=2):
    query_result = collection.query(query_texts=question, n_results=n_results)
    return [doc for sublist in query_result["documents"] for doc in sublist]

def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )
    messages = [
        {
            "role" : "system",
            "content" : prompt 
        },
        {
            "role" : "user",
            "content" : question
        }
    ]
    response = ollama.chat(model='llama3.2', messages=messages)
    return response

if __name__ == '__main__':
    chroma_persistent_client = chromadb.PersistentClient(path='./news_db')
    # embedding API: https://ollama.com/library/all-minilm
    ollama_embed_func = embedding_functions.OllamaEmbeddingFunction(url='http://localhost:11434/api/embeddings', model_name='all-minilm')
    collection_name = 'documents_for_qa'
    collection = chroma_persistent_client.get_or_create_collection(name=collection_name, embedding_function=ollama_embed_func)
    
    print("Enter an Option:")
    print("1. Create database from text files")
    print("2. Query from existing database")
    input_str = input('Enter 1 o 2: ')
    
    if input_str == '1':
        # Database creation
        doc_list = get_document_list('./data/new_articles')
        print(f"loaded {len(doc_list)}")
        # Split documents
        print("====== Split Documents into chunks ========")
        chunked_documents = []
        for doc in doc_list:
            chunks = split_text(doc['text'])
            for chunk in chunks:
                chunked_documents.append({'id': doc['id'],
                                        'text': chunk})
        print(f"Total chunks: {len(chunked_documents)}")
        # Generate embedding of chunk of documents
        print("========= Genereting embeddings ... =====")
        for chunk_doc in chunked_documents:
            chunk_doc['embedding'] = ollama_embed_func(chunk_doc['text'])
        
        ## add chunked documents to db
        print("========= Adding documents and embbedings to db ... =====")
        for chunk_doc in chunked_documents:
            collection.upsert(ids=chunk_doc['id'], documents=chunk_doc['text'], embeddings=chunk_doc['embedding'])
        print(f'{collection.count()} documents added to the db')
    elif input_str == '2':
        # query from existing data base
        
        while True:
            print("============= QUESTION ======================")
            prompt = input("Ask a question or type quit: ")
            if prompt == 'quit':
                break
            else:
                relevant_docs = query_documents(prompt, n_results=5)
                answer = generate_response(prompt, [])
                print("============= ANSWER ======================")
                print(answer['message']['content'])
    else:
        print("Invalid option terminate program")