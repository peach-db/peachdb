from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ContinueConversationRequest(_message.Message):
    __slots__ = ["bot_id", "conversation_id", "query"]
    BOT_ID_FIELD_NUMBER: _ClassVar[int]
    CONVERSATION_ID_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    bot_id: str
    conversation_id: str
    query: str
    def __init__(self, bot_id: _Optional[str] = ..., conversation_id: _Optional[str] = ..., query: _Optional[str] = ...) -> None: ...

class ContinueConversationResponse(_message.Message):
    __slots__ = ["response"]
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    response: str
    def __init__(self, response: _Optional[str] = ...) -> None: ...

class CreateBotRequest(_message.Message):
    __slots__ = ["bot_id", "documents", "system_prompt"]
    BOT_ID_FIELD_NUMBER: _ClassVar[int]
    DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    SYSTEM_PROMPT_FIELD_NUMBER: _ClassVar[int]
    bot_id: str
    documents: _containers.RepeatedScalarFieldContainer[str]
    system_prompt: str
    def __init__(self, bot_id: _Optional[str] = ..., system_prompt: _Optional[str] = ..., documents: _Optional[_Iterable[str]] = ...) -> None: ...

class CreateBotResponse(_message.Message):
    __slots__ = ["status"]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: str
    def __init__(self, status: _Optional[str] = ...) -> None: ...

class CreateConversationRequest(_message.Message):
    __slots__ = ["bot_id", "query"]
    BOT_ID_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    bot_id: str
    query: str
    def __init__(self, bot_id: _Optional[str] = ..., query: _Optional[str] = ...) -> None: ...

class CreateConversationResponse(_message.Message):
    __slots__ = ["conversation_id", "response"]
    CONVERSATION_ID_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    conversation_id: str
    response: str
    def __init__(self, conversation_id: _Optional[str] = ..., response: _Optional[str] = ...) -> None: ...
