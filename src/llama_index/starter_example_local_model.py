from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama
'''
embedding model (133M) : https://huggingface.co/BAAI/bge-small-en-v1.5
'''
# Load data and build an index#
documents = SimpleDirectoryReader("example_data").load_data()

# bge embedding model
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

# ollama
Settings.llm = Ollama(model="mistral", request_timeout=30.0)

index = VectorStoreIndex.from_documents(
    documents,
)


# Query your data#
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)