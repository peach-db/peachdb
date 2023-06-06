"""
PeachDB Library
"""
import abc

# import asyncio
import os
import shelve
import shutil

# from pathlib import Path
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
from peachdb.constants import BLOB_STORE, SHELVE_DB
from peachdb.embedder import EmbeddingProcessor
from peachdb.embedder.utils import Modality, is_s3_uri


class QueryResponse(BaseModel):
    ids: list
    distances: list
    metadata: List[dict]


class _Base(abc.ABC):
    @abc.abstractmethod
    def query(self, text: str, top_k: int = 5):
        pass


class PeachDB(_Base):
    def __init__(
        self,
        project_name,
        embedding_generator: str = "sentence_transformer_L12",
        distance_metric: str = "cosine",
        embedding_backend: str = "exact_cpu",
    ):
        PeachDB._validate_embedding_generator(embedding_generator)
        PeachDB._validate_distance_metric(distance_metric)
        PeachDB._validate_embedding_backend(embedding_backend)
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

        self._db = None

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

    def query(self, text: str, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        assert text and self._db

        if self._db is None:
            self._get_db_backend()

        ids, distances = self._db.process_query(text, top_k)
        metadata = self._db.fetch_metadata(ids)
        return ids, distances, metadata

    def _get_db_backend(self):
        # TODO: our embeddings now have "embeddings_text", "embeddings_audio", "embeddings_image". Adjust for this.
        with shelve.open(SHELVE_DB) as shelve_db:
            assert (
                len(shelve_db[self._project_name]["upsertion_logs"]) == 1
            ), "Only one upsertion per project is supported at this time."

            # TODO: ensure that the info lives here as expected!
            last_upsertion = shelve_db[self._project_name]["upsertion_logs"][-1]

        self._db = get_backend(
            embedding_generator=self._embedding_generator,
            embedding_backend=self._embedding_backend,
            distance_metric=self._distance_metric,
            embeddings_dir=last_upsertion["embeddings_dir"],
            metadata_path=last_upsertion["metadata_path"],
            id_column_name=last_upsertion["id_column_name"],
        )

    def upsert_text(
        self,
        csv_path: str,
        column_to_embed: str,
        id_column_name: str,
        embeddings_output_s3_bucket_uri: Optional[str] = None,
        max_rows: Optional[int] = None,
    ):
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
            embedding_model_name=self._embedding_generator,
            project_name=self._project_name,
            s3_bucket=embeddings_output_s3_bucket_uri,
            modality=Modality.TEXT,
        )

        processor.process()

        with shelve.open(SHELVE_DB) as shelve_db:
            shelve_db[self._project_name]["upsertion_logs"].append(
                {
                    "embeddings_dir": processor.embeddings_output_dir,
                    "metadata_path": csv_path,
                    "column_to_embed": column_to_embed,
                    "id_column_name": id_column_name,
                    "max_rows": max_rows,
                    "embeddings_output_s3_bucket_uri": embeddings_output_s3_bucket_uri,
                    "modality": str(Modality.TEXT),
                }
            )

    def upsert_audio(
        self,
        csv_path: str,
        column_to_embed: str,
        id_column_name: str,
        embeddings_output_s3_bucket_uri: Optional[str] = None,
        max_rows: Optional[int] = None,
    ):
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
            embedding_model_name=self._embedding_generator,
            project_name=self._project_name,
            s3_bucket=embeddings_output_s3_bucket_uri,
            modality=Modality.AUDIO,
        )

        processor.process()

        with shelve.open(SHELVE_DB) as shelve_db:
            shelve_db[self._project_name]["upsertion_logs"].append(
                {
                    "embeddings_dir": processor.embeddings_output_dir,
                    "metadata_path": csv_path,
                    "column_to_embed": column_to_embed,
                    "id_column_name": id_column_name,
                    "max_rows": max_rows,
                    "embeddings_output_s3_bucket_uri": embeddings_output_s3_bucket_uri,
                    "modality": str(Modality.TEXT),
                }
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
        supported_engines = ["sentence_transformer_L12", "imagebind"]
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
