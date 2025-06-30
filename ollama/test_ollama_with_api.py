import requests
import json

# Path of Ollama api endpoint, ollama should be running in the back end 
# more info at: https://github.com/ollama/ollama?tab=readme-ov-file#rest-api
url = 'http://localhost:11434/api/generate'

data = {
    "model" : "llama3.2",
    "prompt" : "How should I get a girlfriend?"
}

request = requests.post(url, json=data, stream=True)

# Check response status
if request.status_code == 200:
    print("Response: ", end=" ", flush=True)
    # Iterate over streaming response
    for line in request.iter_lines():
        if line:
            # Decode the response and parse the json
            decoded_line = line.decode('utf-8')
            result = json.loads(decoded_line)
            # Get the generated text from the response
            generated_text = result.get("response", "")
            print(generated_text, end="", flush=True)
else:
    print("Error:", request.status_code, request.text)
    