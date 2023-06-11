# Bot

Endpoints to manage bots, and conversation with bots. A "bot" is a chatbot that answers the users questions over a list of strings (called "document"s) by performing embedding-based search to inject the most relevant documents into the prompt.

## Endpoints

1. POST /create-bot
2. POST /create-conversation
3. POST /continue-conversation

## 1. Create a bot 
`POST /create-bot`

Takes a list of strings (documents) that the question answering bot will answer questions about.

### Parameters

- `bot_id` (string): A unique ID for the bot that will be used to interact with this bot.
- `system_prompt` (string): The prompt injected at the beginning of any conversation. Could be used to inject information about what the whole collection of documents is about. 
- `documents` (list of strings): The bot will answer questions about these documents. On user interaction, a portion of the documents related to the users query will be used to answer their questions.

### Returns

<!-- TODO -->

### Example

#### Input

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

#### Output

<!-- TODO -->

## 2. Start a conversation
`POST /create-conversation`

Given a bot (that abstracts documents and an agent over it), this end-point starts a conversations with a given user query/question. The query/question is used to decide which subset of the documents will also be used to reply to any further input received by the user as part of this conversation.

To continue the conversation, a `conversation_id` is returned which can be used with the `continue-conversation` end-point.

### Parameters

1. `bot_id`: The unique bot ID used in `/create-bot`.
2. `query`: The query from the user who wants to interact with the bot.

### Returns

<!-- TODO -->

### Example

#### Input
```json
{
    "bot_id": "postman_3",
    "query": "Where did the convention convene?"
}
```

#### Output

<!-- TODO -->

## 3. Continue a conversation
`POST /continue-conversation`

Continues a user conversation started in `create-conversation`. As more queries as sent to this endpoint, the state of the previous queries is used and maintained.

### Parameters

1. `bot_id`: The unique bot ID used in /create-bot.
2. `conversation_id`: The ID returned by `/create-conversation` endpoint.
3. `query`: End-user query to continue the conversation.

### Returns

<!-- TODO -->

### Example

#### Input

```json
{
    "bot_id": "postman_3",
    "conversation_id": "e79204a6-a458-4710-9e83-2d5fa58fbf52",
    "query": "Who was the vice presidential nominee?"
}
```

#### Output

<!-- TODO -->