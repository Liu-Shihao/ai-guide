from langchain_community.embeddings import HuggingFaceEmbeddings
'''
%pip install --upgrade --quiet  langchain sentence_transformers
https://python.langchain.com/docs/integrations/text_embedding/huggingfacehub
'''
# embeddings = HuggingFaceEmbeddings()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
text = "This is a test document."

query_result = embeddings.embed_query(text)
# doc_result = embeddings.embed_documents([text])
print(query_result[:3])