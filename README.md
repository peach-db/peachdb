<h3 align="center">Managed backend for memory-augmented Q&A chatbots</h3>
<h4 align="center">Infra for building chatbots that can truly do back-and-forth and learn user preferences over time</h4>

Easily build chatbots that:
- Are quick to ship by abstracting away computing/storing embeddings & Q&A logic.
- Gather new context, if required, on every back-and-forth of a conversation.
- Learn and use user preferences across conversations.
- Help you improve the dataset used via easy observability.
- Can better retrieve relevant information by finetuning the embeddings model.


# Endpoints

We provide the following endpoints:
1. [HTTPS / non-streaming](https://github.com/peach-db/peachdb/blob/master/README.md#https--non-streaming)
2. [gRPC / streaming](https://github.com/peach-db/peachdb/blob/master/README.md#grpc--streaming)


## HTTPS / non-streaming

Endpoints to manage bots, and conversation with bots. A "bot" is a chatbot that answers the users questions over a list of strings (called "document"s) by performing embedding-based search to inject the most relevant documents into the prompt.

1. POST /create-bot
2. POST /create-conversation
3. POST /continue-conversation

### 1. Create a bot 
`POST /create-bot`

Takes a list of strings (documents) that the question answering bot will answer questions about.

#### Parameters

- `bot_id` (string): A unique ID for the bot that will be used to interact with this bot.
- `system_prompt` (string): The prompt injected at the beginning of any conversation. Could be used to inject information about what the whole collection of documents is about. 
- `documents` (list of strings): The bot will answer questions about these documents. On user interaction, a portion of the documents related to the users query will be used to answer their questions.

#### Returns

- 200 status code on successful bot creation.

##### Errors

- 400:
  - OpenAI server overload.
- 500:
  - Unknown error. Please contact us!

#### Example

##### Input

```json
{
    "bot_id": "postman_3",
    "system_prompt": "Please answer questions about the 1880 Greenback Party National Convention.",
    "documents": [
        "The 1880 Greenback Party National Convention convened in Chicago from June 9 to June 11 to select presidential and vice presidential nominees and write a party platform for the Greenback Party in the United States presidential election of 1880. Delegates chose James B. Weaver of Iowa for President and Barzillai J. Chambers of Texas for Vice President.",
        "The Greenback Party was a newcomer to the political scene in 1880 having arisen, mostly in the nation's West and South, as a response to the economic depression that followed the Panic of 1873. During the Civil War, Congress had authorized greenbacks, a form of money redeemable in government bonds, rather than in then-traditional gold. After the war, many Democrats and Republicans in the East sought to return to the gold standard, and the government began to withdraw greenbacks from circulation. The reduction of the money supply, combined with the economic depression, made life harder for debtors, farmers, and industrial laborers; the Greenback Party hoped to draw support from these groups."
    ]
}
```

##### Output

Returns a 200 status code with message "Bot created successfully."

### 2. Start a conversation
`POST /create-conversation`

Given a bot (that abstracts documents and an agent over it), this end-point starts a conversations with a given user query/question. The query/question is used to decide which subset of the documents will also be used to reply to any further input received by the user as part of this conversation.

To continue the conversation, a `conversation_id` is returned which can be used with the `continue-conversation` end-point.

#### Parameters

1. `bot_id`: The unique bot ID used in `/create-bot`.
2. `query`: The query from the user who wants to interact with the bot.

#### Returns

1. `conversation_id`: A unique ID for a conversation with a given user and start query. Can be used to continue the conversation while preserving context from all intermediary back-and-forths.
2. `response`: The return from the chatbot for the given query.

##### Errors

- 400:
    - `bot_id` not specified
    - `query` not specified
    - OpenAI server overloaded
- 500:
    - Unknown error. Please contact us!

#### Example

##### Input
```json
{
    "bot_id": "postman_3",
    "query": "Where did the convention convene?"
}
```

##### Output

```json
{
    "conversation_id": "e79204a6-a458-4710-9e83-2d5fa58fbf52",
    "response": "The 1880 Greenback Party National Convention convened in Chicago."
}
```

### 3. Continue a conversation
`POST /continue-conversation`

Continues a user conversation started in `create-conversation`. As more queries as sent to this endpoint, the state of the previous queries is used and maintained.

#### Parameters

1. `bot_id`: The unique bot ID used in /create-bot.
2. `conversation_id`: The ID returned by `/create-conversation` endpoint.
3. `query`: End-user query to continue the conversation.

#### Returns

1. `response`: Response to the given user query taking all previous interactions into account.

##### Errors

- 400:
    - `bot_id` not provided
    - `conversation_id` not provided
    - `query` not provided
    - Wrong conversation ID provided.
    - OpenAI servers overloaded
- 500:
    - Unknown error. Please contact us!
#### Example

##### Input

```json
{
    "bot_id": "postman_3",
    "conversation_id": "e79204a6-a458-4710-9e83-2d5fa58fbf52",
    "query": "Who was the vice presidential nominee?"
}
```

##### Output

```json
{
    "response": "The vice presidential nominee for the Greenback Party in the 1880 United States presidential election was Barzillai J. Chambers of Texas."
}
```

## gRPC / streaming

There is an active server running behing `1.tcp.ngrok.io:24448` with the gRPC proto file as specified in `peachdb_grpc/api.proto`.

There are 3 RPCs:

1. `CreateBot` - corresponding to `/create-bot`.
2. `CreateConversation` - corresponding to `/create-conversation` but streams the output.
3. `ContinueConversation` - corresponding to `/continue-conversation` but streams the output.

# Get Involved
We welcome PR contributors and ideas for how to improve the project.

# Special Thanks
To [Modal](https://modal.com/), [DuckDB](https://github.com/duckdb/duckdb) & [pyngrok](https://pypi.org/project/pyngrok/) for developing wonderful services
