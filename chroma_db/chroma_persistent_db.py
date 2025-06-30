import chromadb
from chromadb.utils import embedding_functions

# emb function
default_emb_func = embedding_functions.DefaultEmbeddingFunction()
# create collection
persistent_client = chromadb.PersistentClient(path='./db/chroma_persistent')

collection = persistent_client.get_or_create_collection("my_collection", embedding_function=default_emb_func)

my_collection = [
    {"id" : "Doc1", "text": "I'm the Lizard King, I can do anything!"},
    {"id" : "Doc2", "text": "Hello, I love you won't to tell me your name?"},
    {"id" : "Doc3", "text": "All our lives we sweat and save, building for a shallow grave"},
    {"id" : "Doc4", "text": "No eternal reward will forgive us now for wasting the dawn"}
]

for doc in my_collection:
    collection.upsert(ids=doc['id'], documents=doc['text'])
query = "Hello, I'm a backdoor man"

results = collection.query(query_texts=query, n_results=2)
print(results)
