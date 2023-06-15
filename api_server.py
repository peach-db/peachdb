import asyncio
from typing import AsyncIterable

import grpc  # type: ignore
import openai

import api_pb2
import api_pb2_grpc


class BotServiceServicer(api_pb2_grpc.BotServiceServicer):
    def CreateBot(self, request: api_pb2.CreateBotRequest, context) -> api_pb2.CreateBotResponse:
        # Implement your logic here for creating a bot
        print(request)
        # TODO: implement error handling.
        return api_pb2.CreateBotResponse(status='Bot created successfully')

    async def CreateConversation(self, request: api_pb2.CreateConversationRequest, context) -> AsyncIterable[api_pb2.CreateConversationResponse]:
        # Implement your logic here for creating a conversation
        print(request)
        conversation_id = '123456'
        # responses = ['Response from chatbot 1', 'Response from chatbot 2']
        # yield api_pb2.CreateConversationResponse(conversation_id=conversation_id, response=responses[0])
        # for response in responses[1:]:
        #     await asyncio.sleep(5)
        #     yield api_pb2.CreateConversationResponse(conversation_id=conversation_id, response=response)
        async for resp in await openai.ChatCompletion.acreate(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello, World!"}], stream=True):
            print(resp)
            delta = resp.choices[0].delta
            print(delta)
            if "content" in delta:
                yield api_pb2.CreateConversationResponse(conversation_id=conversation_id, response=delta["content"])
                

    async def ContinueConversation(self, request: api_pb2.ContinueConversationRequest, context) -> AsyncIterable[api_pb2.ContinueConversationResponse]:
        # Implement your logic here for continuing a conversation
        responses = ['Continued response from chatbot 1', 'Continued response from chatbot 2']
        for response in responses:
            yield api_pb2.ContinueConversationResponse(response=response)
            await asyncio.sleep(0.1)


async def serve() -> None:
    server = grpc.aio.server()
    api_pb2_grpc.add_BotServiceServicer_to_server(BotServiceServicer(), server)
    server.add_insecure_port('[::]:50051')
    await server.start()
    await server.wait_for_termination()
    
if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(serve())