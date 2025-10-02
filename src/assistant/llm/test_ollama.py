from ollama_client import OllamaClient

llm = OllamaClient(model="mistral:latest")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Summarize this in one sentence: The patient has high blood pressure and elevated cholesterol."}
]

reply = llm.chat(messages)
print("Model reply:", repr(reply))
