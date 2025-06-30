import json
import ollama

def get_ollama_chat_response(model, messages):
    response = ollama.chat(model = model, messages=messages)
    return response

def get_initial_message():
    return [{'role': 'system', 
             'content': 'You are an efficient assitant'
            }]

def save_conversation_into_json(messages):
    with open('conversation.json', 'w') as json_file:
        json.dump(messages, json_file)
        
def load_conversation_messages():
    with open('conversation.json', 'r') as json_file:
        return json.loads(json_file)
    

if __name__ == '__main__':
    print("============= Chat bot Session, type quit to stop ========================")
    print("Ask or tell me something")
    messages = get_initial_message()
    iterations = 0
    
    while True:
        prompt = input('You: ')
        if iterations > 0:
            messages = load_conversation_messages()
        messages.append({'role': 'user', 
                         'content': f"{prompt}"
                         })
        if prompt.lower() == 'quit':
            messages = []
            save_conversation_into_json(messages)
            break
        response = get_ollama_chat_response(model = 'llama3.2', messages = messages)
        print('Response: ', response['message']['content'])
        print("\n")
        # add last response to messages list
        messages.append({'role': 'assistant', 
                         'content': f"{response['message']['content']}"
                         })
        save_conversation_into_json(messages)
        