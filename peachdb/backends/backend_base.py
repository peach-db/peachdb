import abc
import os

import duckdb
import numpy as np
import pandas as pd
from rich import print

from peachdb.embedder.models.multimodal_imagebind import ImageBindModel
from peachdb.embedder.models.sentence_transformer import SentenceTransformerModel
from peachdb.embedder.utils import S3File, S3Folder, is_s3_uri


class BackendBase(abc.ABC):
    def __init__(
        self,
        embeddings_dir: str,
        metadata_path: str,
        embedding_generator: str,
        distance_metric: str,
        id_column_name: str,
    ):
        self._distance_metric = distance_metric
        self._id_column_name = id_column_name
        self._metadata_filepath = self._get_metadata_filepath(metadata_path)

        self._embeddings, self._ids = self._get_embeddings(embeddings_dir)
        if len(set(self._ids)) != len(self._ids):
            raise ValueError("Duplicate ids found in the embeddings file.")

        if embedding_generator == "sentence_transformer_L12":
            self._encoder = SentenceTransformerModel()
        elif embedding_generator == "imagebind":
            self._encoder = ImageBindModel()
        else:
            raise ValueError(f"Unknown embedding generator: {embedding_generator}")

    @abc.abstractmethod
    def _process_query(self, query_embedding, top_k: int = 5) -> tuple:
        pass

    def process_query(self, query, top_k: int = 5) -> tuple:
        print("Embedding query...")
        query_embedding = self._encoder.encode(texts=[query], batch_size=1, show_progress_bar=True)

        return self._process_query(query_embedding, top_k)

    def fetch_metadata(self, ids) -> pd.DataFrame:
        print("Fetching metadata...")

        data = duckdb.read_csv(self._metadata_filepath)
        id_str = " OR ".join([f"{self._id_column_name} = {id}" for id in ids])
        metadata = duckdb.sql(f"SELECT * FROM data WHERE {id_str}").df()

        return metadata

    def _get_embeddings(self, embeddings_dir: str):
        if not is_s3_uri(embeddings_dir):
            return self._load_embeddings(embeddings_dir)

        print("[bold]Downloading calculated embeddings...[/bold]")
        with S3Folder(embeddings_dir) as tmp_local_embeddings_dir:
            return self._load_embeddings(tmp_local_embeddings_dir)

    def _load_embeddings(self, embeddings_dir: str) -> tuple:
        """Loads and preprocesses the embeddings from a parquet file."""
        assert os.path.exists(embeddings_dir)

        print("[bold]Loading embeddings from parquet file...[/bold]")
        df = pd.read_parquet(embeddings_dir, "pyarrow")

        print("[bold]Converting embeddings to numpy array...[/bold]")
        embeddings = np.array(df["embeddings"].values.tolist()).astype("float32")
        ids = np.array(df["ids"].values.tolist()).astype("int64")
        return embeddings, ids

    def _get_metadata_filepath(self, metadata_path: str) -> str:
        if not is_s3_uri(metadata_path):
            return metadata_path

        print("[bold]Downloading metadata file...[/bold]")
        self._metadata_fileref = S3File(metadata_path)
        return self._metadata_fileref.download()

    def cleanup(self):
        if is_s3_uri(self._metadata_path):
            self._metadata_fileref.cleanup()

        if is_s3_uri(self._embeddings_dir):
            self._embeddings_dir.cleanup()
