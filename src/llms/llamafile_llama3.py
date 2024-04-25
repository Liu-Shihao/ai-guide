from langchain_community.llms.llamafile import Llamafile

"""
chmod +x Meta-Llama-3-8B-Instruct.Q4_0.llamafile

# Start the model server. Listens at http://localhost:8080 by default.
./Meta-Llama-3-8B-Instruct.Q4_0.llamafile --server --nobrowser
"""
model = "/Users/liushihao/Downloads/model/llamafiles/Meta-Llama-3-8B-Instruct.Q4_0.llamafile"
llm = Llamafile()

print(llm.invoke("你好."))