from flask import Flask, request, jsonify, Response, render_template
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)

llm = ChatOllama(model="llama3")
prompt = ChatPromptTemplate.from_template("{input}")
chain = prompt | llm | StrOutputParser()
def generate_data(input):
    prompt = {"input": input}
    for chunks in chain.stream(prompt):
        # print(chunks)
        yield f"{chunks}"
@app.route('/chatbot', methods=['POST'])
def chatbot():
    if not request.json:
        return jsonify({'error': 'No JSON data received'}), 400
    message = request.json.get('message')
    print(f"接收到用户输入：{message}")
    return Response(generate_data(message), mimetype='text/plain')

@app.route('/')
def index():
    return render_template('index1.html')
if __name__ == '__main__':
    app.run(debug=True,
            port=8080)
