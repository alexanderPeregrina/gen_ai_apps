from chromadb.utils import embedding_functions
import numpy as np

default_embedding_func = embedding_functions.DefaultEmbeddingFunction()

prompt = "I hope it can  continue just a little while longer"

result_emb = default_embedding_func(prompt)

print(type(result_emb))
print(prompt.strip())
print(len(prompt.strip()))
print(np.shape(result_emb))