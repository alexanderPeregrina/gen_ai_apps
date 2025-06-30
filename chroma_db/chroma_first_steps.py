import chromadb

""" This script generate embedding vector data base for a collection of documents and 
    query a sentence retrieves the most similar documents to the query"""
# create client
client = chromadb.Client()

# collection
colection_name = "test_collection"

collection = client.get_or_create_collection(colection_name)

documents = [
    {"id": "doc1", "text": "Hello World!"},
    {"id": "doc2", "text": "I'm the lizard King, I can do anything"},
    {"id": "doc3", "text": "Hello, Goodbye"},
    {"id" : "doc4", "text": "My eyes have seen you!"}
    
]

for document in documents:
    collection.upsert(document["id"], documents=[document["text"]])

query = "Hello, good morning"

results = collection.query(query_texts=query, n_results=2)

print(results)
