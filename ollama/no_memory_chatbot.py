import ollama

print("============= Chat bot Session, type quit to stop ========================")
while True:
    prompt = input('Ask or tell me something: ')
    if prompt.lower() == 'quit':
        break
    response = ollama.chat(model = 'llama3.2',
                        messages=[{'role': 'user', 'content' : f'{prompt}'}],
                        stream=True)

    for chunk in response:
        print(chunk['message']['content'], end= "", flush=True)
    print("\n")