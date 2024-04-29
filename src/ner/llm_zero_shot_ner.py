from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat = ChatOllama(model="llama3")

template = """
你是一个NER模型，仔细思考，你有充足的时间进行严谨的思考，然后提取出文本中的Date、Amount、Payee相关信息。
例如：I want transfer $1 to ajay on 2024-04-30. 提取出 AMOUNT:1$, PAYEE:ajay,DATE:2024-04-30输出。
如果没有提供，请不要猜测，可以提供部分相关信息。
"""

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            template,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | chat

if __name__ == '__main__':
    result = chain.invoke(
        {
            "messages": [
                HumanMessage(
                    content="I want transfer $1 to ajay on 2024-04-30"
                )
            ],
        }
    )
    print(result)
