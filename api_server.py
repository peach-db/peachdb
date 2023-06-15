import asyncio
from typing import AsyncIterable

import grpc  # type: ignore
import openai

import api_pb2
import api_pb2_grpc
from peachdb.bots.qa import BadBotInputError, ConversationNotFoundError, QABot


class BotServiceServicer(api_pb2_grpc.BotServiceServicer):
    def CreateBot(self, request: api_pb2.CreateBotRequest, context) -> api_pb2.CreateBotResponse:
        try:
            try:
                bot = QABot(
                    bot_id=request.bot_id,
                    system_prompt=request.system_prompt,
                    llm_model_name=request.llm_model_name if request.HasField("llm_model_name") else "gpt-3.5-turbo",
                    embedding_model=request.embedding_model_name
                    if request.HasField("embedding_model_name")
                    else "openai_ada",
                )
            except BadBotInputError as e:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(str(e))
                return api_pb2.CreateBotResponse()

            try:
                bot.add_data(documents=request.documents)
                return api_pb2.CreateBotResponse(status="Bot created successfully.")
            except openai.error.RateLimitError:
                context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
                context.set_details("OpenAI's servers are currently overloaded. Please try again later.")
                return api_pb2.CreateBotResponse()
            except openai.error.AuthenticationError:
                context.set_code(grpc.StatusCode.UNAUTHENTICATED)
                context.set_details("There's been an authentication error. Please contact the team.")
                return api_pb2.CreateBotResponse()
        except Exception as e:
            context.set_code(grpc.StatusCode.UNKNOWN)
            context.set_details("An unknown error occurred. Please contact the team.")
            return api_pb2.CreateBotResponse()

    # TODO: add streaming support below.
    # async def CreateConversation(
    #     self, request: api_pb2.CreateConversationRequest, context
    # ) -> AsyncIterable[api_pb2.CreateConversationResponse]:
    #     # Implement your logic here for creating a conversation
    #     print(request)
    #     conversation_id = "123456"
    #     # responses = ['Response from chatbot 1', 'Response from chatbot 2']
    #     # yield api_pb2.CreateConversationResponse(conversation_id=conversation_id, response=responses[0])
    #     # for response in responses[1:]:
    #     #     await asyncio.sleep(5)
    #     #     yield api_pb2.CreateConversationResponse(conversation_id=conversation_id, response=response)
    #     async for resp in await openai.ChatCompletion.acreate(
    #         model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello, World!"}], stream=True
    #     ):
    #         print(resp)
    #         delta = resp.choices[0].delta
    #         print(delta)
    #         if "content" in delta:
    #             yield api_pb2.CreateConversationResponse(conversation_id=conversation_id, response=delta["content"])

    async def CreateConversation(
        self, request: api_pb2.CreateConversationRequest, context
    ) -> AsyncIterable[api_pb2.CreateConversationResponse]:
        try:
            if not request.bot_id:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("bot_id must be specified.")
                yield api_pb2.CreateConversationResponse()
                return

            if not request.query:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("query must be specified.")
                yield api_pb2.CreateConversationResponse()
                return

            bot_id = request.bot_id
            query = request.query

            bot = QABot(bot_id=bot_id)
            try:
                # TODO: make this async given above code.
                cid, response = bot.create_conversation_with_query(query=query)
            except openai.error.RateLimitError:
                context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
                context.set_details("OpenAI's servers are currently overloaded. Please try again later.")
                yield api_pb2.CreateConversationResponse()
                return

            print(cid, response)
            yield api_pb2.CreateConversationResponse(conversation_id=cid, response=response)
            return
        except Exception as e:
            context.set_code(grpc.StatusCode.UNKNOWN)
            context.set_details("An unknown error occurred. Please contact the team.")
            yield api_pb2.CreateConversationResponse()
            return

    async def ContinueConversation(
        self, request: api_pb2.ContinueConversationRequest, context
    ) -> AsyncIterable[api_pb2.ContinueConversationResponse]:
        try:
            if not request.bot_id:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("bot_id must be specified.")
                yield api_pb2.ContinueConversationResponse()
                return

            if not request.conversation_id:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("conversation_id must be specified.")
                yield api_pb2.ContinueConversationResponse()
                return

            if not request.query:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("query must be specified.")
                yield api_pb2.ContinueConversationResponse()
                return

            bot_id = request.bot_id
            conversation_id = request.conversation_id
            query = request.query

            bot = QABot(bot_id=bot_id)
            try:
                response = bot.continue_conversation_with_query(conversation_id=conversation_id, query=query)
            except ConversationNotFoundError:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Conversation not found. Please check `conversation_id`")
                yield api_pb2.ContinueConversationResponse()
                return
            except openai.error.RateLimitError:
                context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
                context.set_details("OpenAI's servers are currently overloaded. Please try again later.")
                yield api_pb2.ContinueConversationResponse()
                return

            # TODO: add streaming
            yield api_pb2.ContinueConversationResponse(response=response)
            return
        except Exception as e:
            context.set_code(grpc.StatusCode.UNKNOWN)
            context.set_details("An unknown error occurred. Please contact the team.")
            yield api_pb2.ContinueConversationResponse()
            return


async def serve() -> None:
    server = grpc.aio.server()
    api_pb2_grpc.add_BotServiceServicer_to_server(BotServiceServicer(), server)
    server.add_insecure_port("[::]:50051")
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(serve())
