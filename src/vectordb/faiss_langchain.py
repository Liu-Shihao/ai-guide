# Uncomment the following line if you need to initialize FAISS with no AVX2 optimization
# os.environ['FAISS_NO_AVX2'] = '1'

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

'''

'''
loader = TextLoader("../../modules/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)
print(db.index.ntotal)

# Querying
query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)

# As a Retriever
retriever = db.as_retriever()
docs = retriever.invoke(query)
print(docs[0].page_content)

# Similarity Search with score
docs_and_scores = db.similarity_search_with_score(query)

# Saving and loading
db.save_local("faiss_index")

new_db = FAISS.load_local("faiss_index", embeddings)

docs = new_db.similarity_search(query)

# Merging
# You can also merge two FAISS vectorstores
db1 = FAISS.from_texts(["foo"], embeddings)
db2 = FAISS.from_texts(["bar"], embeddings)

db1.docstore._dict
db2.docstore._dict
db1.merge_from(db2)
db1.docstore._dict