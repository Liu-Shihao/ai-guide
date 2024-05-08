from flask import Flask, Response, render_template
import time

from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)

llm = ChatOllama(model="llama3")
prompt = ChatPromptTemplate.from_template("{input}")
chain = prompt | llm | StrOutputParser()
def generate_data():
    prompt = {"input": input}
    for chunks in chain.stream(prompt):
        # print(chunks)
        yield chunks
    # for i in range(10):
    #     # 模拟生成数据
    #     time.sleep(1)
    #     yield f"Data {i}\n"

@app.route('/stream_data')
def stream_data():
    # 使用生成器函数生成数据流
    return Response(generate_data(), mimetype='text/plain')

@app.route('/')
def index():
    return render_template('http_stream.html')
if __name__ == '__main__':
    app.run(debug=True,
            port=8080)
