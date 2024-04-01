from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
'''
pip install beautifulsoup4
pip install faiss-gpu (for CUDA supported GPU) or pip install faiss-cpu
pip install sentence-transformers
pip install langchainhub

https://python.langchain.com/docs/use_cases/question_answering/quickstart

https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
1.llm

2.embedding

3.vector store

4.retriever
'''


loader = WebBaseLoader("https://gorilla.cs.berkeley.edu/blogs/9_raft.html")

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

all_splits = text_splitter.split_documents(docs)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")

vectorstore = FAISS.from_documents(all_splits, embeddings)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})


# retrieved_docs = retriever.invoke("What is the RAFT?")
# print(retrieved_docs[0].page_content)

llm = ChatOllama(model="llama2")

prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

for chunk in rag_chain.stream("What is the RAFT?"):
    print(chunk, end="", flush=True)

'''
The RAFT (RAG Fine-tuning) is a method to adapt a pre-trained language model (LLM) to a specific domain or task by fine-tuning it on a dataset that contains questions, context, and answers from the target domain. The goal of RAFT is to improve the LLM's performance on a particular set of questions or tasks by tailoring it to the specific context and requirements of the target domain.

In the context of this blog post, RAFT is being used to adapt a pre-trained LLM (Llama2-7B) to answer questions related to a specific medical topic (Medical) and general knowledge (General-knowledge). The RAFT model is trained on top of the base model Llama2-7B from Meta and Microsoft AI Studio, and the training data includes questions, context, and answers from these domains.

The RAFT evaluation involves retaining an oracle document (di*) along with distractor documents (dk-1) for a fraction (1-P) of the questions in the dataset, and fine-tuning the language model using standard supervised training (SFT) technique. The figure below illustrates the high-level design principal for RAFT.

In summary, RAFT is a recipe for adapting a pre-trained LLM to a specific domain or task by fine-tuning it on a dataset that contains questions, context, and answers from the target domain. The goal of RAFT is to improve the LLM's performance on a particular set of questions or tasks by tailoring it to the specific context and requirements of the target domain.

'''