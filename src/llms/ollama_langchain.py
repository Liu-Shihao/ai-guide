"""

https://python.langchain.com/docs/integrations/llms/ollama#usage
https://python.langchain.com/docs/integrations/chat/ollama

ollama run llama2

ollama pull llama2:7b-chat

curl http://localhost:11434/api/generate -d '{
  "model": "llama2",
  "prompt":"Why is the sky blue?"
}'

On Mac, the models will be download to ~/.ollama/models

On Linux (or WSL), the models will be stored at /usr/share/ollama/.ollama/models
"""


from langchain_community.llms import Ollama

llm = Ollama(model="llama2")

query = "Tell me a joke"

# llm.invoke(query)

for chunks in llm.stream(query):
    print(chunks)


