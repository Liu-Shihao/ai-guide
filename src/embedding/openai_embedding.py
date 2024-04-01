from langchain_openai import OpenAIEmbeddings
'''

'''

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# embeddings_1024 = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)
text = "This is a test document."
query_result = embeddings.embed_query(text)