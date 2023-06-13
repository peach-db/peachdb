import os
import shelve
import tempfile
from uuid import uuid4

import openai
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import Response
from pydantic import BaseModel
from pyngrok import ngrok  # type: ignore
from rich import print

from peachdb import EmptyNamespace, PeachDB
from peachdb.bots.qa import ConversationNotFoundError, QABot
from peachdb.constants import SHELVE_DB

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EMBEDDING_GENERATOR = "openai_ada"
EMBEDDING_BACKEND = "exact_cpu"


def _validate_data(texts, ids, metadatas_list):
    assert all(isinstance(text, str) for text in texts), "All texts must be strings"
    assert all(isinstance(i, str) for i in ids), "All IDs must be strings"
    assert all(isinstance(m, dict) for m in metadatas_list), "All metadata must be dicts"
    assert len(set([str(x.keys()) for x in metadatas_list])) == 1, "All metadata must have the same keys"


def _validate_metadata_key_names_dont_conflict(metadatas_dict, namespace):
    assert "texts" not in metadatas_dict.keys(), "Metadata cannot contain a key called 'texts'"
    assert "ids" not in metadatas_dict.keys(), "Metadata cannot contain a key called 'ids'"

    if namespace is not None:
        assert "namespace" not in metadatas_dict.keys(), "Metadata cannot contain a key called 'namespace'"


def _process_input_data(request_json: dict) -> pd.DataFrame:
    input_data = request_json
    namespace = request_json.get("namespace", None)

    # TODO: make metadata optional?

    # tuples of (id: str, text: list[float], metadata: dict[str, str]])
    data = input_data["data"]

    assert len(set([len(d) for d in data])) == 1, "All data must have the same length"
    if len(data[0]) == 3:
        # we got (ids, texts, metadata)
        pass
    elif len(data[0]) == 2:
        # we got (ids, texts)
        data = [(d[0], d[1], {}) for d in data]
    else:
        raise ValueError("Data must be of the form (ids, texts) or (ids, texts, metadata)")

    ids = [str(d[0]) for d in data]
    texts = [d[1] for d in data]
    metadatas_list: list = [d[2] for d in data]

    _validate_data(texts, ids, metadatas_list)

    # convert metadata from input format to one we can create a dataframe from.
    metadatas_dict: dict[str, list] = {
        key: [metadata[key] for metadata in metadatas_list] for key in metadatas_list[0].keys()
    }

    _validate_metadata_key_names_dont_conflict(metadatas_dict, namespace)
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


def http_error_handle(fn):
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            return Response(content="An unknown error occured. Please contact the PeachDB team.", status_code=500)

    return wrapper


# TODO: we need an init that figures out which model to use?
@app.post("/upsert-text")
# TODO: make below decorator work.
# @http_error_handle()
async def upsert_handler(request: Request):
    """
    Takes texts as input rather than vectors (unlike Pinecone).
    """
    input_data = await request.json()

    project_name = input_data.get("project_name", None)
    if project_name is None:
        return Response(content="Project name must be specified.", status_code=400)

    peach_db = PeachDB(
        project_name=project_name,
        embedding_generator=EMBEDDING_GENERATOR,
        embedding_backend=EMBEDDING_BACKEND,
    )
    new_data_df = _process_input_data(input_data)
    namespace = input_data.get("namespace", None)

    with shelve.open(SHELVE_DB) as shelve_db:
        project_info = shelve_db.get(project_name, None)
        assert project_info is not None, "Project not found"
        assert not project_info["lock"], "Some other process is currently reading/writing to this project"
        # TODO: replace with a lock we can await.
        project_info["lock"] = True
        shelve_db[project_name] = project_info

    if os.path.exists(project_info["exp_compound_csv_path"]):
        data_df = pd.read_csv(project_info["exp_compound_csv_path"])

        if namespace is None:
            data_df_namespace = data_df[data_df["namespace"].isnull()]
        else:
            data_df_namespace = data_df[data_df["namespace"] == namespace]

        # Check for intersection between the "ids" column of data_df and new_data_df
        # TODO: add support on backend for string ids.
        if len(set(data_df_namespace["ids"].apply(str)).intersection(set(new_data_df["ids"].apply(str)))) != 0:
            with shelve.open(SHELVE_DB) as shelve_db:
                project_info = shelve_db.get(project_name, None)
                project_info["lock"] = False
                shelve_db[project_name] = project_info

            return Response(
                content="New data contains IDs that already exist in the database for this namespace. This is not allowed.",
                status_code=400,
            )

    # We use unique csv_name to avoid conflicts in the stored data.
    with tempfile.NamedTemporaryFile(suffix=f"{uuid4()}.csv") as tmp:
        # TODO: what happens if ids' conflict?
        new_data_df.to_csv(tmp.name, index=False)  # TODO: check it won't cause an override.

        peach_db.upsert_text(
            csv_path=tmp.name,
            column_to_embed="texts",
            id_column_name="ids",
            # TODO: below is manually set, this might cause issues!
            embeddings_output_s3_bucket_uri="s3://metavoice-vector-db/deployed_solution/",
            namespace=namespace,
        )

    if os.path.exists(project_info["exp_compound_csv_path"]):
        # Update the data_df with the new data, and save to disk.
        data_df = pd.concat([data_df, new_data_df], ignore_index=True)
        data_df.to_csv(project_info["exp_compound_csv_path"], index=False)
    else:
        new_data_df.to_csv(project_info["exp_compound_csv_path"], index=False)

    # release lock
    with shelve.open(SHELVE_DB) as shelve_db:
        project_info = shelve_db.get(project_name, None)
        project_info["lock"] = False
        shelve_db[project_name] = project_info


@app.get("/query")
async def query_embeddings_handler(request: Request):
    data = await request.json()

    project_name = data.get("project_name", None)
    if project_name is None:
        return Response(content="Project name must be specified.", status_code=400)

    peach_db = PeachDB(
        project_name=project_name,
        embedding_generator=EMBEDDING_GENERATOR,
        embedding_backend=EMBEDDING_BACKEND,
    )

    text = data["text"]
    top_k = int(data.get("top_k", 5))
    namespace = data.get("namespace", None)

    try:
        ids, distances, metadata = peach_db.query(query_input=text, modality="text", namespace=namespace, top_k=top_k)
    except EmptyNamespace:
        return Response(content="Empty namespace.", status_code=400)

    result = []
    # TODO: we're aligning distances, ids, and metadata from different sources which could cause bugs.
    # Fix this.
    for id, dist in zip(ids, distances):
        values = metadata[metadata["ids"] == id].values[0]
        columns = list(metadata.columns)
        columns = [
            c for c in columns if c != "namespace"
        ]  # Causes an error with JSON encoding when "namespace" is None and ends up as NaN here.
        result_dict = {columns[i]: values[i] for i in range(len(columns))}
        result_dict["distance"] = dist
        result.append(result_dict)

    return {"result": result}


@app.post("/create-bot")
async def create_bot_handler(request: Request):
    try:
        request_json = await request.json()

        bot = QABot(
            bot_id=request_json["bot_id"],
            system_prompt=request_json["system_prompt"],
            llm_model_name=request_json["llm_model_name"] if "llm_model_name" in request_json else "gpt-3.5-turbo",
            embedding_model=request_json["embedding_model_name"]
            if "embedding_model_name" in request_json
            else "openai_ada",
        )

        try:
            bot.add_data(documents=request_json["documents"])
            return Response(content="Bot created successfully.", status_code=200)
        except openai.error.RateLimitError:
            return Response(
                content="OpenAI's server are currently overloaded. Please try again later.", status_code=400
            )
    except Exception as e:
        return Response(content="An unknown error occured. Please contact the team.", status_code=500)


@app.post("/create-conversation")
async def create_conversation_handler(request: Request):
    try:
        request_json = await request.json()

        if "bot_id" not in request_json:
            return Response(content="bot_id must be specified.", status_code=400)

        if "query" not in request_json:
            return Response(content="query must be specified.", status_code=400)

        bot_id = request_json["bot_id"]
        query = request_json["query"]

        bot = QABot(bot_id=bot_id)
        try:
            cid, response = bot.create_conversation_with_query(query=query)
        except openai.error.RateLimitError:
            return Response(
                content="OpenAI's server are currently overloaded. Please try again later.", status_code=400
            )

        return {
            "conversation_id": cid,
            "response": response,
        }
    except Exception as e:
        return Response(content="An unknown error occured. Please contact the team.", status_code=500)


@app.post("/continue-conversation")
async def continue_conversation_handler(request: Request):
    try:
        request_json = await request.json()

        if "bot_id" not in request_json:
            return Response(content="bot_id must be specified.", status_code=400)
        if "conversation_id" not in request_json:
            return Response(content="conversation_id must be specified.", status_code=400)
        if "query" not in request_json:
            return Response(content="query must be specified.", status_code=400)

        bot_id = request_json["bot_id"]
        conversation_id = request_json["conversation_id"]
        query = request_json["query"]

        bot = QABot(bot_id=bot_id)
        try:
            response = bot.continue_conversation_with_query(conversation_id=conversation_id, query=query)
        except ConversationNotFoundError:
            return Response(content="Conversation not found. Please check `conversation_id`", status_code=400)
        except openai.error.RateLimitError:
            return Response(
                content="OpenAI's server are currently overloaded. Please try again later.", status_code=400
            )

        return {
            "response": response,
        }
    except Exception as e:
        return Response(content="An unknown error occured. Please contact the team.", status_code=500)


if __name__ == "__main__":
    port = 8000
    url = ngrok.connect(port)
    print(f"[green]Public URL: {url}[/green]")
    uvicorn.run("deploy_api:app", host="0.0.0.0", port=port)  # , reload=True)
