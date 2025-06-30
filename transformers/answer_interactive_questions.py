from transformers import pipeline, AutoTokenizer

def generate_llm_model():
    generator = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')
    return generator

if __name__ == "__main__":
    text_generator = generate_llm_model()
    print("Interactive questions, enter quit to exit")

    context = """
            The Transformers library by Hugging Face provides thousands of pre-trained models to perform tasks on texts such as classification, information extraction, question answering, summarization, translation, and more.
        """

    while True:
        question = input("Insert a question: ")
        if question.lower() == "quit":
            break
        output = text_generator(question= question, context = context)
        print(f"ANSWER: {output['answer']}")