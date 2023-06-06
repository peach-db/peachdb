"""
PeachDB Library
"""
import abc
import os
import shelve
import shutil
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pyngrok import ngrok  # type: ignore
from rich import print
from rich.prompt import Prompt

from peachdb.backends import get_backend
from peachdb.backends.backend_base import BackendBase
from peachdb.constants import BLOB_STORE, SHELVE_DB
from peachdb.embedder import EmbeddingProcessor
from peachdb.embedder.utils import Modality, is_s3_uri
from peachdb.validators import (
    validate_columns,
    validate_csv_path,
    validate_distance_metric,
    validate_embedding_backend,
    validate_embedding_generator,
)


class QueryResponse(BaseModel):
    ids: list
    distances: list
    metadata: List[dict]


class _Base(abc.ABC):
    @abc.abstractmethod
    def query(self, query_input: str, modality: Modality, store_modality: Optional[Modality] = None, top_k: int = 5):
        pass


class PeachDB(_Base):
    def __init__(
        self,
        project_name,
        embedding_generator: str = "sentence_transformer_L12",
        distance_metric: str = "cosine",
        embedding_backend: str = "exact_cpu",
    ):
        validate_embedding_generator(embedding_generator)
        validate_distance_metric(distance_metric)
        validate_embedding_backend(embedding_backend)
        super().__init__()
        self._project_name = project_name
        self._embedding_generator = embedding_generator
        self._distance_metric = distance_metric
        self._embedding_backend = embedding_backend

        with shelve.open(SHELVE_DB) as shelve_db:
            if self._project_name in shelve_db.keys():
                shelve_db[project_name]["query_logs"].append(
                    {"distance_metric": distance_metric, "embedding_backend": embedding_backend}
                )
            else:
                shelve_db[project_name] = {
                    "embedding_generator": embedding_generator,
                    "query_logs": [{"distance_metric": distance_metric, "embedding_backend": embedding_backend}],
                    "upsertion_logs": [],
                }
                print(f"[u]PeachDB has been created for project: [bold green]{project_name}[/bold green][/u]")

        self._db: Optional[BackendBase] = None

    def deploy(self):
        if self._db is None:
            self._get_db_backend()

        app = FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/query", response_model=QueryResponse)
        async def query_handler(query_input: str, modality: str, top_k: int = 5):
            if isinstance(modality, str):
                modality = Modality(modality)
            ids, distances, metadata = self.query(query_input, modality=modality, top_k=top_k)
            return {
                "ids": ids.tolist(),
                "distances": distances.tolist(),
                "metadata": metadata.to_dict(orient="records"),
            }

        port = 8000
        url = ngrok.connect(port)
        print(f"[green]Public URL: {url}[/green]")
        try:
            uvicorn.run(app, host="0.0.0.0", port=port)
        except KeyboardInterrupt:
            self._db.cleanup()

    def query(
        self,
        query_input: str,
        modality: str | Modality,
        store_modality: Optional[Modality] = None,
        top_k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        assert query_input and modality
        if isinstance(modality, str):
            modality = Modality(modality)

        if self._db is None:
            self._db = self._get_db_backend()

        ids, distances = self._db.process_query(query=query_input, top_k=top_k, modality=modality)
        metadata = self._db.fetch_metadata(ids)
        return ids, distances, metadata

    # TODO: handle store_modality
    def _get_db_backend(self, store_modality=None):
        with shelve.open(SHELVE_DB) as shelve_db:
            if len(shelve_db[self._project_name]["upsertion_logs"]) < 1:
                raise ValueError("No embeddings! Please upsert data before running your query")

            # TODO: ensure that the info lives here as expected!
            last_upsertion = shelve_db[self._project_name]["upsertion_logs"][-1]

        embeddings_dir = last_upsertion["embeddings_dir"]
        metadata_path = last_upsertion["metadata_path"]
        id_column_name = last_upsertion["id_column_name"]
        # TODO: fix if we have multiple modalities stored.
        store_modality = store_modality if store_modality is not None else Modality(last_upsertion["modality"])

        return get_backend(
            embedding_generator=self._embedding_generator,
            embedding_backend=self._embedding_backend,
            distance_metric=self._distance_metric,
            embeddings_dir=embeddings_dir,
            metadata_path=metadata_path,
            id_column_name=id_column_name,
            modality=store_modality,
        )

    def _upsert(
        self,
        csv_path: str,
        column_to_embed: str,
        id_column_name: str,
        modality: Modality,
        embeddings_output_s3_bucket_uri: Optional[str] = None,
        max_rows: Optional[int] = None,
    ):
        validate_csv_path(csv_path)
        validate_columns(column_to_embed, id_column_name, csv_path)

        if is_s3_uri(csv_path):
            assert (
                embeddings_output_s3_bucket_uri
            ), "Please provide `embeddings_output_s3_bucket_uri` for output embeddings when the `csv_path` is an S3 URI."

            assert is_s3_uri(
                embeddings_output_s3_bucket_uri
            ), f"The provided output_bucket_s3_uri {embeddings_output_s3_bucket_uri} is not a S3 URI"

        shelve_db = shelve.open(SHELVE_DB)
        if len(shelve_db[self._project_name]["upsertion_logs"]) == 1:
            raise ValueError(
                "Only one upsertion is supported at this time. Either continue with this instance, or create a new one using PeachDB.create(...). You can reclaim this `project_name` by called PeachDB.delete(...)"
            )

        processor = EmbeddingProcessor(
            csv_path=csv_path,
            column_to_embed=column_to_embed,
            id_column_name=id_column_name,
            max_rows=max_rows,
            embedding_model_name=self._embedding_generator,
            project_name=self._project_name,
            embeddings_output_s3_bucket_uri=embeddings_output_s3_bucket_uri,
            modality=modality,
        )

        processor.process()

        _save = shelve_db[self._project_name]

        _save["upsertion_logs"].append(
            {
                "embeddings_dir": processor.embeddings_output_dir,
                "metadata_path": csv_path,
                "column_to_embed": column_to_embed,
                "id_column_name": id_column_name,
                "max_rows": max_rows,
                "embeddings_output_s3_bucket_uri": embeddings_output_s3_bucket_uri,
                "modality": modality.value,
            }
        )
        shelve_db[self._project_name] = _save

    def upsert_text(
        self,
        csv_path: str,
        column_to_embed: str,
        id_column_name: str,
        embeddings_output_s3_bucket_uri: Optional[str] = None,
        max_rows: Optional[int] = None,
    ):
        self._upsert(
            csv_path=csv_path,
            column_to_embed=column_to_embed,
            id_column_name=id_column_name,
            embeddings_output_s3_bucket_uri=embeddings_output_s3_bucket_uri,
            max_rows=max_rows,
            modality=Modality.TEXT,
        )

    def upsert_audio(
        self,
        csv_path: str,
        column_to_embed: str,
        id_column_name: str,
        embeddings_output_s3_bucket_uri: Optional[str] = None,
        max_rows: Optional[int] = None,
    ):
        self._upsert(
            csv_path=csv_path,
            column_to_embed=column_to_embed,
            id_column_name=id_column_name,
            embeddings_output_s3_bucket_uri=embeddings_output_s3_bucket_uri,
            max_rows=max_rows,
            modality=Modality.AUDIO,
        )

    def upsert_image(
        self,
        csv_path: str,
        column_to_embed: str,
        id_column_name: str,
        embeddings_output_s3_bucket_uri: Optional[str] = None,
        max_rows: Optional[int] = None,
    ):
        self._upsert(
            csv_path=csv_path,
            column_to_embed=column_to_embed,
            id_column_name=id_column_name,
            embeddings_output_s3_bucket_uri=embeddings_output_s3_bucket_uri,
            max_rows=max_rows,
            modality=Modality.IMAGE,
        )

    @staticmethod
    def delete(project_name: str):
        db = shelve.open(SHELVE_DB)
        if project_name not in db.keys():
            print(f"Project: {project_name} does not exist.")
            return

        answer = Prompt.ask(f"[bold red]Would you like to delete the project: {project_name}? (y/N)[/]")
        delete_project = answer.lower() == "y"
        if delete_project:
            del db[project_name]
            project_dir = os.path.join(BLOB_STORE, project_name)
            if os.path.exists(project_dir):
                shutil.rmtree(project_dir)

        print(f"[green]Successfully deleted project: {project_name}[/]")

    @staticmethod
    def _ensure_unique_project_name(project_name: str):
        with shelve.open(SHELVE_DB) as db:
            assert (
                project_name not in db.keys()
            ), f"The project name '{project_name}' already exists. Please choose a unique name."
