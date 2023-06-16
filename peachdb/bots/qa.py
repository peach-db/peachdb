import dotenv

dotenv.load_dotenv()

import shelve
import tempfile
from typing import Iterator, Optional, Union
from uuid import uuid4

import openai
import pandas as pd

from peachdb import PeachDB
from peachdb.constants import BOTS_DB, CONVERSATIONS_DB, SHELVE_DB


class ConversationNotFoundError(ValueError):
    pass

class UnexpectedGPTRoleResponse(ValueError):
    pass

def _validate_embedding_model(embedding_model: str):
    assert embedding_model in ["openai_ada"]


def _validate_llm_model(llm_model):
    assert llm_model in ["gpt-3.5-turbo", "gpt-4"]


def _process_input_data(namespace, ids: list[str], texts: list[str], metadatas_dict) -> pd.DataFrame:
    assert all(isinstance(text, str) for text in texts), "All texts must be strings"
    assert all(isinstance(i, str) for i in ids), "All IDs must be strings"
    if metadatas_dict is not None:
        assert all(isinstance(m, dict) for m in metadatas_dict), "All metadata must be dicts"
        assert len(set([str(x.keys()) for x in metadatas_dict])) == 1, "All metadata must have the same keys"

        # convert metadata from input format to one we can create a dataframe from.
        metadatas_dict = {key: [metadata[key] for metadata in metadatas_dict] for key in metadatas_dict[0].keys()}

        assert "texts" not in metadatas_dict.keys(), "Metadata cannot contain a key called 'texts'"
        assert "ids" not in metadatas_dict.keys(), "Metadata cannot contain a key called 'ids'"

        if namespace is not None:
            assert "namespace" not in metadatas_dict.keys(), "Metadata cannot contain a key called 'namespace'"
    else:
        metadatas_dict = {}

    if namespace is None:
        metadatas_dict["namespace"] = [None] * len(ids)
    else:
        metadatas_dict["namespace"] = [namespace] * len(ids)

    df = pd.DataFrame(
        data={
            "ids": ids,
            "texts": texts,
            **metadatas_dict,
        }
    )

    return df


def _peachdb_upsert_wrapper(peach_db_instance, peach_db_project_name: str, namespace, ids, texts, metadatas_dict):
    new_data_df = _process_input_data(namespace, ids, texts, metadatas_dict)

    with tempfile.NamedTemporaryFile(suffix=f"{uuid4()}.csv") as tmp:
        new_data_df.to_csv(tmp.name, index=False)  # TODO: check it won't cause an override.

        peach_db_instance.upsert_text(
            csv_path=tmp.name,
            column_to_embed="texts",
            id_column_name="ids",
            # TODO: below is manually set, this might cause issues!
            embeddings_output_s3_bucket_uri="s3://metavoice-vector-db/deployed_solution/",
            namespace=namespace,
        )

    with shelve.open(SHELVE_DB) as db:
        project_info = db[peach_db_project_name]
        new_data_df.to_csv(project_info["exp_compound_csv_path"], index=False)

    return True


class BadBotInputError(ValueError):
    pass


class QABot:
    def __init__(
        self,
        bot_id: str,
        embedding_model: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        with shelve.open(BOTS_DB) as db:
            if bot_id in db:
                if system_prompt is not None:
                    raise BadBotInputError(
                        "System prompt cannot be changed for existing bot. Maybe you want to create a new bot?"
                    )
                if embedding_model is not None:
                    raise BadBotInputError(
                        "Embedding model cannot be changed for existing bot. Maybe you want to create a new bot?"
                    )
                if llm_model_name is not None:
                    raise BadBotInputError(
                        "LLM model cannot be changed for existing bot. Maybe you want to create a new bot?"
                    )
                self._peachdb_project_id = db[bot_id]["peachdb_project_id"]
                self._embedding_model = db[bot_id]["embedding_model"]
                self._llm_model_name = db[bot_id]["llm_model_name"]
                self._system_prompt = db[bot_id]["system_prompt"]
            else:
                if system_prompt is None:
                    raise BadBotInputError("System prompt must be specified for new bot.")
                if embedding_model is None:
                    raise BadBotInputError("Embedding model must be specified for new bot.")
                if llm_model_name is None:
                    raise BadBotInputError("LLM model must be specified for new bot.")

                self._peachdb_project_id = f"{uuid4()}_{bot_id}"
                self._embedding_model = embedding_model
                self._llm_model_name = llm_model_name
                self._system_prompt = system_prompt

                db[bot_id] = {
                    "peachdb_project_id": self._peachdb_project_id,
                    "embedding_model": self._embedding_model,
                    "llm_model_name": self._llm_model_name,
                    "system_prompt": self._system_prompt,
                }

        _validate_embedding_model(self._embedding_model)
        _validate_llm_model(self._llm_model_name)

        self.peach_db = PeachDB(
            project_name=self._peachdb_project_id,
            embedding_generator=self._embedding_model,
        )

        if self._llm_model_name in ["gpt-3.5-turbo", "gpt-4"]:
            self._llm_model = lambda messages, stream: openai.ChatCompletion.create(
                messages=[{"role": "system", "content": self._system_prompt}] + messages,
                model=self._llm_model_name,
                stream=stream,
            )
        else:
            raise ValueError(f"Unknown/Unsupported LLM model: {self._llm_model_name}")

    def add_data(self, documents: list[str]):
        _peachdb_upsert_wrapper(
            peach_db_instance=self.peach_db,
            peach_db_project_name=self._peachdb_project_id,
            namespace=None,
            ids=[str(i) for i in range(len(documents))],
            texts=documents,
            metadatas_dict=None,
        )

    def _llm_response(self, conversation_id: str, messages: list[dict[str, str]], stream: bool = False) -> Union[tuple[str,str], Iterator[tuple[str, str]]]:
        """
        Responds to the given messages with the LLM model. Additionally, it appends to the shelve db the current conversation (After response has been returned from GPT).
        """
        response = self._llm_model(messages=messages, stream=stream)

        if stream:
            response_str = ""

            for resp in response:
                delta = resp.choices[0].delta

                if "role" in delta:
                    if delta.role != "assistant":
                        raise UnexpectedGPTRoleResponse(f"Expected assistant response, got {delta.role} response.")

                if "content" in delta:
                    response_str += delta["content"]
                    yield conversation_id, delta["content"]

                # keep updating shelve with current conversation.
                with shelve.open(CONVERSATIONS_DB) as db:
                    db[conversation_id] = messages + [{"role": "assistant", "content": response_str}]
        else:
            response_message = response.choices[0].message
            if response_message.role != "assistant":
                raise UnexpectedGPTRoleResponse(f"Expected assistant response, got {response_message.role} response.")

            with shelve.open(CONVERSATIONS_DB) as db:
                db[conversation_id] = messages + [response_message]

            return conversation_id, response_message["content"]
        
    def _create_unique_conversation_id(self) -> str:
        # get conversation id not in shelve.
        id = str(uuid4())
        with shelve.open(CONVERSATIONS_DB) as db:
            while id in db:
                id = str(uuid4())
                
        return id
        

    def create_conversation_with_query(
        self, query: str, top_k: int = 3, stream: bool = False
    ) -> Union[tuple[str, str], Iterator[tuple[str, str]]]:
        _, _, context_metadata = self.peach_db.query(query, top_k=top_k, modality="text")
        assert "texts" in context_metadata

        contextual_query = "Use the below snippets to answer the subsequent questions. If the answer can't be found, write \"I don't know.\""
        for text in context_metadata["texts"]:
            contextual_query += f"\n\nSnippet:\n{text}"
        contextual_query += f"\n\nQuestion:{query}"

        # add context to query
        messages = [
            {"role": "user", "content": contextual_query},
        ]
        
        conversation_id = self._create_unique_conversation_id()
                
        if stream:
            for x in self._llm_response(conversation_id, messages, stream=True):
                yield x
        else:
            return self._llm_response(conversation_id, messages, stream=False)
        

    def continue_conversation_with_query(
        self, conversation_id: str, query: str, top_k: int = 3, stream: bool = False
    ) -> Union[str, Iterator[str]]:
        with shelve.open(CONVERSATIONS_DB) as db:
            if conversation_id not in db:
                raise ConversationNotFoundError("Conversation ID not found.")

            messages = db[conversation_id]

        messages.append({"role": "user", "content": query})

        if stream:
            # TODO: fix below type issue.
            for (_, response) in self._llm_response(conversation_id, messages, stream=True): # type: ignore
                yield response
        else:
            _, response = self._llm_response(conversation_id, messages, stream=False)
            return response
