import asyncio

import grpc  # type: ignore

import api_pb2
import api_pb2_grpc


def create_bot(stub: api_pb2_grpc.BotServiceStub):
    repsonse = stub.CreateBot(
        api_pb2.CreateBotRequest(
            bot_id="grpc_test", documents=["Hello", "World"], system_prompt="Answer questions about this document."
        )
    )

    print(repsonse)

async def create_conversation(stub: api_pb2_grpc.BotServiceStub):
    responses = stub.CreateConversation(
        api_pb2.CreateConversationRequest(bot_id="grpc_test", query="What is this document about?")
    )

    async for response in responses:
        print("Create conversation response:")
        print(response)
    
async def continue_conversation(stub: api_pb2_grpc.BotServiceStub):
    responses = stub.ContinueConversation(
        api_pb2.ContinueConversationRequest(conversation_id="123456", query="What is this document about?")
    )
    
    async for response in responses:
        print("Continue conversation response:")
        print(response)

async def main() -> None:
    # async with grpc.aio.insecure_channel("localhost:50051") as channel:
    async with grpc.aio.insecure_channel("6.tcp.eu.ngrok.io:11380") as channel:
        stub = api_pb2_grpc.BotServiceStub(channel)
        
        await asyncio.gather(
            create_conversation(stub),
            continue_conversation(stub)
        )

if __name__ == "__main__":
    asyncio.run(main())