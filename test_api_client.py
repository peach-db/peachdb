import asyncio

import grpc  # type: ignore

import api_pb2
import api_pb2_grpc


def create_bot(stub: api_pb2_grpc.BotServiceStub):
    repsonse = stub.CreateBot(
        api_pb2.CreateBotRequest(
            bot_id="grpc_test_2", documents=["Hello", "World"], system_prompt="Answer questions about this document."
        )
    )

    return repsonse


async def create_conversation(stub: api_pb2_grpc.BotServiceStub):
    responses = stub.CreateConversation(
        api_pb2.CreateConversationRequest(bot_id="grpc_test_2", query="What is this document about?")
    )
    
    async for response in responses:
        print("Create conversation response:")
        print(response)
    
async def continue_conversation(stub: api_pb2_grpc.BotServiceStub):
    responses = stub.ContinueConversation(
        api_pb2.ContinueConversationRequest(bot_id="grpc_test_2", conversation_id="fae52c34-84c2-4d17-aee3-f7afc1f34261", query="What is this document about?")
    )

    async for response in responses:
        # TODO: make streamable given openAI.
        print("Continue conversation response:")
        print(response)


async def main() -> None:
    async with grpc.aio.insecure_channel("localhost:50051") as channel:
        # async with grpc.aio.insecure_channel("1.tcp.ngrok.io:24448") as channel:
        stub = api_pb2_grpc.BotServiceStub(channel)

        # print(await create_bot(stub))
        # await create_conversation(stub)
        await continue_conversation(stub)

        # await asyncio.gather(
        #     create_conversation(stub),
        #     continue_conversation(stub)
        # )


if __name__ == "__main__":
    asyncio.run(main())
