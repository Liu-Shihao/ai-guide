from langchain_community.embeddings import OllamaEmbeddings

'''
https://python.langchain.com/docs/integrations/text_embedding/ollama
'''
text = "This is a test document."

embeddings = OllamaEmbeddings()
# embeddings = OllamaEmbeddings(model="llama2:7b")

# query_result = embeddings.embed_query(text)
# query_result[:5]

doc_result = embeddings.embed_documents([text])
print(doc_result[0][:5])