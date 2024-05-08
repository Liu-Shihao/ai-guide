import asyncio
import websockets

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
"""
pip install websockets

"""


# llm = ChatOllama(model="llama3")
# prompt = ChatPromptTemplate.from_template("{input}")
# chain = prompt | llm | StrOutputParser()

# async def data_stream(input):
#     prompt = {"input": input}
#     for chunks in chain.stream(prompt):
#         # print(chunks)
#         yield chunks

async def handle_client(websocket, path):
    async for message in websocket:
        print(f"Received message from client: {message}")
        await websocket.send(f"Server received: {message}")

async def main():
    server = await websockets.serve(handle_client, "localhost", 8765)
    print("Server started...")
    await server.wait_closed()

asyncio.run(main())
