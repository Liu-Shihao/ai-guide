from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
'''
https://milvus.io/docs
https://python.langchain.com/docs/integrations/vectorstores/milvus

'''
loader = TextLoader("../../modules/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()


vector_db = Milvus.from_documents(
    docs,
    embeddings,
    connection_args={"host": "127.0.0.1", "port": "19530"},
)

query = "What did the president say about Ketanji Brown Jackson"
docs = vector_db.similarity_search(query)

print(docs[0].page_content)

vector_db = Milvus.from_documents(
    docs,
    embeddings,
    collection_name="collection_1",
    connection_args={"host": "127.0.0.1", "port": "19530"},
)

vector_db = Milvus(
    embeddings,
    connection_args={"host": "127.0.0.1", "port": "19530"},
    collection_name="collection_1",
)