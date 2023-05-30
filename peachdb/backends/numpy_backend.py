import os
from typing import Tuple

import duckdb
import numpy as np
import pandas as pd
from rich import print

from peachdb.backends.backend_base import BackendBase


def _check_dims(query_embed: np.ndarray, embeds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if query_embed.ndim == 1:
        query_embed = query_embed[np.newaxis, :]
    elif query_embed.ndim == 2:
        if query_embed.shape[0] != 1:
            raise ValueError("query_embed should be a vector or a matrix with one row")
    else:
        raise ValueError("query_embed should be a vector or a matrix with one row")

    if embeds.ndim != 2:
        raise ValueError("embeds should be a 2-D matrix")

    return query_embed, embeds


def l2(query_embed: np.ndarray, embeds: np.ndarray) -> np.ndarray:
    """
    Calculate l2 distance between a query embedding and a set of embeddings.
    """
    query_embed, embeds = _check_dims(query_embed, embeds)

    return np.linalg.norm(query_embed - embeds, axis=1)


def cosine(query_embed: np.ndarray, embeds: np.ndarray) -> np.ndarray:
    """
    Can be used to compute cosine "distance" between any number of query embeddings
    and a set of embeddings.
    result[i, j] = 1 - np.dot(query_embed[i], embeds[j])
    """
    query_embed, embeds = _check_dims(query_embed, embeds)

    return (1 - (query_embed @ embeds.T) / (np.linalg.norm(query_embed, axis=1) * np.linalg.norm(embeds, axis=1)))[0]


class NumpyBackend(BackendBase):
    def _process_query(self, query_embedding, top_k: int = 5):
        """Compute query embedding, calculate distance of query embedding and get top k."""
        print("Calculating distances...")
        distances = (
            l2(query_embedding, self._embeddings)
            if self._distance_metric == "l2"
            else cosine(query_embedding, self._embeddings)
        )

        print("Getting top results...")
        results = np.argsort(distances)[:top_k]
        return self._ids[results], distances[results]


if __name__ == "__main__":
    import scipy.spatial.distance as scipy_distance
    from sentence_transformers.util import cos_sim as st_cos_sim

    for dim in [3, 384, 1536]:
        a = np.random.rand(dim)
        b = np.random.rand(3, dim)

        # cosine
        cosine_result = cosine(a, b)
        for i in range(b.shape[0]):
            np.testing.assert_allclose(scipy_distance.cosine(a, b[i]), cosine_result[i])
            np.testing.assert_allclose(1 - st_cos_sim(a, b[i]).numpy(), cosine_result[i])

        # l2
        l2_result = l2(a, b)
        for i in range(b.shape[0]):
            np.testing.assert_allclose(scipy_distance.euclidean(a, b[i]), l2_result[i])
