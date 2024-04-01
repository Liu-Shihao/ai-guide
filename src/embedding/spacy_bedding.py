from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings

'''
%pip install --upgrade --quiet  spacy
'''

embedder = SpacyEmbeddings(model_name="en_core_web_sm")
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs.",
    "How vexingly quick daft zebras jump!",
    "Bright vixens jump; dozy fowl quack.",
]

embeddings = embedder.embed_documents(texts)
for i, embedding in enumerate(embeddings):
    print(f"Embedding for document {i+1}: {embedding}")

query = "Quick foxes and lazy dogs."
query_embedding = embedder.embed_query(query)
print(f"Embedding for query: {query_embedding}")