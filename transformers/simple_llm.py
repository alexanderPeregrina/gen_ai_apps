from transformers import pipeline, AutoTokenizer

def create_simple_llm():
    """
        Creates a simple LLM using GPT-2 model (smallest version)
    """
    generator = pipeline("text-generation", model="distilgpt2")
    
    return generator

generator = create_simple_llm()
output = generator("Jim Morrison was", max_length=150, num_return_sequences=1, do_sample = True, truncation = True,  temperature = 0.2)
print(output)
