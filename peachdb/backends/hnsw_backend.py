from typing import Tuple

import hnswlib
import numpy as np
from rich import print

from peachdb.backends.backend_base import BackendBase


class HNSWBackend(BackendBase):
    def __init__(
        self,
        embeddings_dir: str,
        metadata_path: str,
        embedding_generator: str,
        distance_metric: str,
        id_column_name: str,
    ):
        super().__init__(
            embeddings_dir=embeddings_dir,
            metadata_path=metadata_path,
            embedding_generator=embedding_generator,
            distance_metric=distance_metric,
            id_column_name=id_column_name,
        )
        if self._embeddings.ndim != 2:
            raise ValueError("embeddings should be a 2-D matrix")

        self._dim = self._embeddings.shape[1]
        # create hnsw index.
        self._hnsw_index = hnswlib.Index(space=self._distance_metric, dim=self._dim)

        self._max_elements = self._embeddings.shape[0]
        # initialise index.
        # TODO: fix to support multiple upserts. (#multiple-upserts)
        self._hnsw_index.init_index(
            max_elements=self._max_elements,
            ef_construction=min(200, self._embeddings.shape[0]),  # default param
            M=16,  # default param
            random_seed=100,
        )

        # add data points to index.
        print("[bold]Adding data points to index...[/bold]")
        self._hnsw_index.add_items(self._embeddings, self._ids)

        # set hnsw ef param
        self._hnsw_index.set_ef(max(200, self._embeddings.shape[0]))

    def _process_query(self, query_embedding, top_k: int = 5):
        """Compute query embedding, calculate distance of query embedding and get top k."""
        if query_embedding.ndim != 1 and not (query_embedding.ndim == 2 and query_embedding.shape[0] == 1):
            raise ValueError("query_embedding should be a vector or a matrix with one row")

        print("Getting top results...")
        labels, distances = self._hnsw_index.knn_query(query_embedding, k=top_k)
        return labels[0], distances[0]
