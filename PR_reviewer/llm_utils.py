import ollama
import re

class LLM_reviewer:
    def __init__(self, model_name):
        self.model = model_name

    def review_document(self, document, differences, file_status):
        document_for_llm =  self.__add_line_numbers_to_document(document)
        review_non_new_document_prompt = \
            f"""You are an expert software reviewer. I will provide you with:
                - The new version of a code document (with line numbers included).
                - A unified diff string showing the differences compared to the previous version.

                Your task is to:

                1. Focus strictly on the differences. Do not review unchanged code.
                2. Provide **feedback only if the changes introduce issues or opportunities for improvement** in clarity, readability, maintainability, performance, or language usage.
                3. Suggest **refactoring opportunities** where applicable, and explain why they would help.
                4. When suggesting changes, always **specify the exact line numbers** and the involved function, using the format:
                   [Comment on line <number>]:
                5. If the changes are acceptable and you find nothing to improve, respond only with:
                   *“The changes are good to me.”*

                ⚠️ Important rules:
                - Do not provide positive comments inline with the code.
                - Only give comments if the changes are truly worthy of implementation.
                - Reserve any positive or general feedback for the **Final Assessment** section.

                At the end of your review, provide a general comment for the current file using this format:
                - **Final Assessment**: Highlight overall strengths and any potential improvements. Positive remarks are allowed here.

                Here is the new version of the code document:
                {document_for_llm}

                Here are the differences:
                {differences}

            """.strip()
        
        review_new_document_prompt = f""" You are an expert software reviewer. I will provide you with:
            - A newly added code document (with line numbers included).
            - There is no previous version of this file, so you must review the entire document.

        Your task is to:

        1. Review the entire code document for issues or opportunities for improvement in clarity, readability, maintainability, performance, or language usage.
        2. Suggest **refactoring opportunities** where applicable, and explain why they would help.
        3. When suggesting changes, always **specify the exact line numbers** and the involved function, using the format:
           [Comment on line <number>]:
        4. If the code is acceptable and you find nothing to improve, respond only with:
           *“The code is good to me.”*

        ⚠️ Important rules:
        - Do not provide positive comments inline with the code.
        - Only give comments if the suggestions are truly worthy of implementation.
        - Reserve any positive or general feedback for the **Final Assessment** section.

        At the end of your review, provide a general comment for the current file using this format:
        - **Final Assessment**: Highlight overall strengths and any potential improvements. Positive remarks are allowed here.

        Here is the new version of the code document:
        {document_for_llm}
        """.strip()
        if file_status == 'added':
            prompt = review_new_document_prompt
        else:
            prompt = review_non_new_document_prompt
        print("Reviewing document...")

        response = ollama.chat(model = self.model, messages=[{'role': 'user', 'content' : f'{prompt}'}],
                               stream=False, options={'temperature': 0})
            
        return self.__get_formatted_comments(response['message']['content'])
    
    def generate_general_comment(self, comments):
        prompt = f"""You are an assistant that summarizes GitHub pull request review comments.

                    Input:
                    {comments}

                    Task:
                    - Provide a concise summary of the main points raised in the comments.
                    - Group related feedback together (e.g., style issues, logic concerns, documentation requests).
                    - Highlight any blockers or required changes before approval.
                    - Use clear, neutral language without repeating the original text verbatim.
                    - Output should be in bullet points for readability.""".strip()
        
        response = ollama.chat(model = self.model, messages=[{'role': 'user', 'content' : f'{prompt}'}],
                                                   stream=False, options={'temperature': 0})
        
        return response['message']['content']
    
    def __add_line_numbers_to_document(self, text):
        lines_list = []
        for i, line in enumerate(text.splitlines()):
            lines_list.append(f"{i+1}: " + line)

        return "\n".join(lines_list)
    
    def __get_formatted_comments(self, response):
        pattern = r"\[Comment on line (\d+)\]:\s*(.*?)(?=\n\[Comment on line|\n---|\n\*\*Final Assessment|\Z)"
        final_pattern = r"\*\*Final Assessment\*\*:(.*)"
        # Extract inline comments
        comments = re.findall(pattern, string=response, flags=re.DOTALL)
        comment_dict = {}
        for line, comment in comments:
            comment_dict[line] = comment.strip()

        # Extract final assessment
        final = re.search(pattern=final_pattern, string=response, flags=re.DOTALL)
        if final:
            comment_dict['final'] = final.group(1).strip()

        return comment_dict