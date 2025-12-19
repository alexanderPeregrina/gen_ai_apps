import ollama
import re

SOURCE_FILE_NAME = "sort_elements.cpp"
def generate_source_file(response):
    with open('sorting_elements.cpp', 'w') as cpp_code:
        match = re.search(r'```cpp\s*(.*?)\s*```', response, re.IGNORECASE | re.DOTALL)
        if match:
            # starting from 6th index to remove ```cpp, and 3 index less at the end to remove ```
            source_code = response[match.start() + 6: match.end()-3]
            with open(SOURCE_FILE_NAME, 'w') as cpp_code:
                cpp_code.write(source_code)
                print(f"Source file generated with name {SOURCE_FILE_NAME}")


def extract_explanation(response):
    pattern = r"\*\*Explanation\*\*"
    match = re.search(pattern, response)
    if match:
        return response[match.start()::]
    return ""

if __name__ == "__main__":

    prompt = """Write a c++17 program that sorts the elements in vector that contains the following elements: {10, 20, 5, 6, 8, 9, -4, 8, 1}
                    print the output, provide a single proposal, also provide an explanation of the code after the code, use ```cpp ``` to enclose the code.
                    Use **Explanation** to provide the explanation"""
    

    print("This program generates the code of a c++17 program that sorts elements in an vector in ascending order")
    print("Generating code...")
    response = ollama.chat(model = 'qwen3-coder:480b-cloud',
                            messages=[{'role': 'user', 'content' : f'{prompt}'}],
                            stream=False, options={'temperature': 0})
    
    generate_source_file(response['message']['content'])

    print(extract_explanation(response['message']['content']))



    
