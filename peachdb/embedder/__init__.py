import os
import shutil
from collections import namedtuple
from typing import List, Optional, Type

import duckdb
import pyarrow.parquet as pq  # type: ignore
from rich import print

import peachdb.embedder.containers.base
from peachdb.constants import BLOB_STORE
from peachdb.embedder.utils import Modality, S3File, is_s3_uri

Chunk = namedtuple("Chunk", ["texts_or_paths", "ids"])


# TODO: split into two separate classes LocalEmbeddingsProcessor & S3EmbeddingsProcessor
class EmbeddingProcessor:
    def __init__(
        self,
        csv_path: str,
        column_to_embed: str,
        id_column_name: str,
        embedding_model_name: str,
        project_name: str,
        modality: Modality,
        embeddings_output_s3_bucket_uri: Optional[str] = None,
        max_rows: Optional[int] = None,
    ):
        self._csv_path = csv_path
        self._column_to_embed = column_to_embed
        self._id_column_name = id_column_name
        self._embedding_model_name = embedding_model_name
        self._max_rows = max_rows
        self._project_name = project_name
        self._embeddings_output_s3_bucket_uri = (
            embeddings_output_s3_bucket_uri.strip("/") + "/" if embeddings_output_s3_bucket_uri else None
        )
        self._modality = modality

        if self._embedding_model_name == "sentence_transformer_L12":
            from peachdb.embedder.containers.sentence_transformer import SentenceTransformerEmbedder, sbert_stub

            self._embedding_model: Type[
                peachdb.embedder.containers.base.EmbeddingModelBase
            ] = SentenceTransformerEmbedder
            self._embedding_model_stub = sbert_stub
            self._embedding_model_chunk_size = 10000
        elif self._embedding_model_name == "imagebind":
            from peachdb.embedder.containers.multimodal_imagebind import ImageBindEmbdedder, imagebind_stub

            self._embedding_model = ImageBindEmbdedder
            self._embedding_model_stub = imagebind_stub
            self._embedding_model_chunk_size = 1000
        else:
            raise ValueError(f"Invalid embedding model name: {self._embedding_model_name}")

    @property
    def embeddings_output_dir(self):
        if is_s3_uri(self._csv_path):
            return f"{self._embeddings_output_s3_bucket_uri}{self._project_name}/embeddings"

        dir = f"{BLOB_STORE}/{self._project_name}/embeddings"
        os.makedirs(dir, exist_ok=True)
        return dir

    def process(self):
        dataset = self._download_and_read_dataset(self._csv_path)

        print("[bold]Chunking data into batches...[/bold]")
        chunked_data = self._chunk_data(dataset)

        print("[bold]Running embedding model on each chunk in parallel...[/bold]")
        self._run_model(chunked_data)

    def _download_and_read_dataset(self, csv_path: str) -> str:
        """Supports a local/s3 reference to the csv formatted dataset"""
        if not is_s3_uri(self._csv_path):
            print("[bold]Loading data into memory...[/bold]")
            # local ref has been provided. make a copy within .peachdb for persistence
            project_blob_dir = f"{BLOB_STORE}/{self._project_name}"
            os.makedirs(project_blob_dir, exist_ok=True)

            fname = self._csv_path.split("/")[-1]
            dataset_path = f"{project_blob_dir}/{fname}"
            shutil.copy2(self._csv_path, dataset_path)
            return self._read_dataset(dataset_path)
        else:
            print("[bold]Downloading data from S3...[/bold]")
            with S3File(self._csv_path) as downloaded_dataset:
                print("[bold]Loading data into memory...[/bold]")
                return self._read_dataset(downloaded_dataset)

    def _read_dataset(self, dataset_path: str):
        data = duckdb.read_csv(dataset_path, header=True)
        sql_query = f"SELECT {self._column_to_embed}, {self._id_column_name} FROM data"
        if self._max_rows:
            sql_query += f" LIMIT {self._max_rows}"
        return duckdb.sql(sql_query).fetchall()  # NOTE: takes 2 mins 10 seconds for a large dataset

    def _chunk_data(self, fetched_data) -> List[Chunk]:
        chunk_size = self._embedding_model_chunk_size
        chunked_data = [fetched_data[i : i + chunk_size] for i in range(0, len(fetched_data), chunk_size)]
        print(f"[bold]...{len(fetched_data)} rows were split into {len(chunked_data)} chunks[/bold]")

        inputs = []

        for chunk in chunked_data:
            texts_or_paths = []
            ids = []
            for text_or_path, id in chunk:
                texts_or_paths.append(text_or_path)
                ids.append(id)
            inputs += [Chunk(texts_or_paths, ids)]

        return inputs

    def _run_model(self, chunks: List[Chunk]):
        with self._embedding_model_stub.run():
            st = self._embedding_model()

            fname = self._csv_path.split("/")[-1].split(".")[0]
            input_tuples = [
                # expected: (ids, output_path, texts, audio_paths, image_paths, show_progress)
                (
                    chunk.ids,
                    f"{self.embeddings_output_dir}/{fname}_{idx}_{self._embedding_model_name}.parquet",
                    # TODO: enable support of using multiple modalities at the same time here (#multi-modality)
                    chunk.texts_or_paths if self._modality == Modality.TEXT else None,
                    chunk.texts_or_paths if self._modality == Modality.AUDIO else None,
                    chunk.texts_or_paths if self._modality == Modality.IMAGE else None,
                    True,
                )
                for idx, chunk in enumerate(chunks)
            ]
            results = list(st.calculate_embeddings.starmap(input_tuples))  # type: ignore

            if not is_s3_uri(self._csv_path):
                for idx, result in enumerate(results):
                    pq.write_table(
                        result, f"{self.embeddings_output_dir}/{fname}_{idx}_{self._embedding_model_name}.parquet"
                    )
