from transformers import pipeline, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('distilgpt2')

text = "Ich arbeite nicht, ich liebe dich"

tokens = tokenizer.encode(text)

decode = tokenizer.decode(tokens)

print(f"tokens: {tokens}")
print(f"decode: {decode}")