import asyncio
import websockets

async def send_message():
    async with websockets.connect("ws://localhost:8765") as websocket:
        while True:
            message = input("Enter message to send to server: ")
            await websocket.send(message)
            response = await websocket.recv()
            print(f"Received from server: {response}")

asyncio.run(send_message())
