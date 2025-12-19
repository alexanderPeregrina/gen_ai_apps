import ollama

# This is the maximum number of token the model qwen3-coder:480b-cloud can receive
# To check this info for other models use: ollama show <my_model> in the ollama's CLI
DEFAULT_CONTEXT_LENGTH = 262144
FILE_PATH_TO_ANALYZE = "../PDF_RAG_system/rag_pdf_system.py"

def extract_document(source_file):
    text = ""
    with open(file=source_file, mode='r', encoding='utf-8') as file:
        text = file.read()
    return text

if __name__ == "__main__":

    document = extract_document(FILE_PATH_TO_ANALYZE)
    prompt = f"""You are an expert software reviewer. I will provide you with a code document. 
                Your task is to:
                
                1. Review the code carefully and provide **feedback** on clarity, readability, and maintainability.
                2. Suggest **potential improvements** or optimizations, including performance enhancements, simplifications, or better use of language features.
                3. Recommend **refactoring opportunities** where applicable, and explain why they would help.
                4. When suggesting changes, Make sure to always **specify the exact lines of code** and the involved function, as your comments will be commented on the code
                using the line references. Show the previous and improved version. This also apply for typo fixes.
                5. If the code is already well-written and you don't see anything to improve, simply respond: 
                   *“The changes are good to me.”*
                
                Please structure your response with clear sections:
                - **Issues / Improvements**, Here you put the changes suggested changes of code, be specific, list all changes (use numbers).
                - **Recommended changes**, Here you can put general comments of potential improvements.
                - **Final Assessment**, Highlight strengths and the potential improvements.
                
                Here is the code document:
                {document}"""
    print("Reviewing document...")
    response = ollama.chat(model = 'qwen3-coder:480b-cloud',
                            messages=[{'role': 'user', 'content' : f'{prompt}'}],
                            stream=False, options={'temperature': 0})
    
    print(response['message']['content'])