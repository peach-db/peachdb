"""
PeachDB Library
"""
import abc
import asyncio
import os
import shelve
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pyngrok import ngrok
from rich import print
from rich.prompt import Prompt

from peachdb.backends import get_backend
from peachdb.constants import BLOB_STORE, SHELVE_DB
from peachdb.embedder import EmbeddingProcessor
from peachdb.embedder.utils import is_s3_uri


class QueryResponse(BaseModel):
    ids: list
    distances: list
    metadata: List[dict]


class _Base(abc.ABC):
    @abc.abstractmethod
    def query(self, text: str, top_k: int = 5):
        pass

    @abc.abstractmethod
    def upsert(self, text: str):
        pass


class PeachDB(_Base):
    def __init__(
        self,
        project_name,
        embedding_generator: str = "sentence_transformer_L12",
        distance_metric: str = "cosine",
        embedding_backend: str = "exact_cpu",
    ):
        super().__init__()
        self._project_name = project_name
        self._embeddings_dir = None
        self._metadata_path = None

        with shelve.open(SHELVE_DB) as shelve_db:
            if self._project_name in shelve_db.keys():
                self._embeddings_dir = shelve_db[self._project_name]["embeddings_dir"]
                self._metadata_path = shelve_db[self._project_name]["metadata_path"]
                self._id_column_name = shelve_db[self._project_name]["id_column_name"]
            else:
                raise ValueError(
                    f"Project name: {project_name} not found. Creating an empty PeachDB is not supported yet, please use PeachDB.create"
                )

        self._db = get_backend(
            embedding_generator=embedding_generator,
            embedding_backend=embedding_backend,
            distance_metric=distance_metric,
            embeddings_dir=self._embeddings_dir,
            metadata_path=self._metadata_path,
            id_column_name=self._id_column_name,
        )

    def deploy(self):
        app = FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/query", response_model=QueryResponse)
        async def get_query_handler(text: str, top_k: int = 5):
            ids, distances, metadata = self.query(text, top_k=top_k)
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

    def query(self, text: str, top_k: int = 5) -> Tuple[list, list, pd.DataFrame]:
        assert text
        ids, distances = self._db.process_query(text, top_k)
        metadata = self._db.fetch_metadata(ids)
        return ids, distances, metadata

    def upsert(self, text: str):
        raise NotImplementedError

    @staticmethod
    def create(
        project_name: str,
        csv_path: str,
        column_to_embed: str,
        id_column_name: str,
        embeddings_output_s3_bucket_uri: Optional[str] = None,
        max_rows: Optional[int] = None,
        embedding_generator: str = "sentence_transformer_L12",
        embedding_backend: str = "exact_cpu",
        distance_metric: str = "cosine",
    ) -> "PeachDB":
        PeachDB._ensure_unique_project_name(project_name)
        PeachDB._validate_embedding_generator(embedding_generator)
        PeachDB._validate_distance_metric(distance_metric)
        PeachDB._validate_embedding_backend(embedding_backend)
        PeachDB._validate_csv_path(csv_path)
        assert column_to_embed
        assert id_column_name

        if is_s3_uri(csv_path):
            assert (
                embeddings_output_s3_bucket_uri
            ), "Please provide `embeddings_output_s3_bucket_uri` for output embeddings when the `csv_path` is an S3 URI."

        processor = EmbeddingProcessor(
            csv_path=csv_path,
            column_to_embed=column_to_embed,
            id_column_name=id_column_name,
            max_rows=max_rows,
            embedding_model_name=embedding_generator,
            project_name=project_name,
            s3_bucket=embeddings_output_s3_bucket_uri,
        )

        processor.process()

        with shelve.open(SHELVE_DB) as db:
            db[project_name] = {
                "metadata_path": csv_path,
                "column_to_embed": column_to_embed,
                "id_column_name": id_column_name,
                "max_rows": max_rows,
                "embedding_generator": embedding_generator,
                "distance_metric": distance_metric,
                "embedding_backend": embedding_backend,
                "embeddings_dir": processor.embeddings_output_dir,
                "embeddings_output_s3_bucket_uri": embeddings_output_s3_bucket_uri,
            }

        print(f"[u]PeachDB has been created for project: [bold green]{project_name}[/bold green][/u]")
        return PeachDB(
            project_name,
            embedding_generator=embedding_generator,
            distance_metric=distance_metric,
            embedding_backend=embedding_backend,
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

    @staticmethod
    def _validate_embedding_generator(engine: str):
        supported_engines = ["sentence_transformer_L12"]
        assert (
            engine in supported_engines
        ), f"Unsupported embedding generator. Currently supported engines are: {', '.join(supported_engines)}"

    @staticmethod
    def _validate_distance_metric(metric: str):
        supported_metrics = ["l2", "cosine"]
        assert (
            metric in supported_metrics
        ), f"Unsupported distance metric. The metric should be one of the following: {', '.join(supported_metrics)}"

    @staticmethod
    def _validate_embedding_backend(backend: str):
        supported_backends = ["exact_cpu", "exact_gpu", "approx"]
        assert (
            backend in supported_backends
        ), f"Unsupported embedding backend. The backend should be one of the following: {', '.join(supported_backends)}"

    @staticmethod
    def _validate_csv_path(csv_path: str):
        assert csv_path, "csv_path parameter is missing. Please provide a valid local file path or an S3 URI."

        if not is_s3_uri(csv_path):
            assert os.path.exists(
                csv_path
            ), f"The provided csv_path '{csv_path}' does not exist. Please ensure that the path is either a valid local file path or an S3 URI (e.g., s3://path/to/csv)."
