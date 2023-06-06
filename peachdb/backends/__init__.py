from typing import Dict, Union

from peachdb.backends.backend_base import BackendBase, BackendConfig
from peachdb.backends.hnsw_backend import HNSWBackend
from peachdb.backends.numpy_backend import NumpyBackend
from peachdb.backends.torch_backend import TorchBackend
from peachdb.embedder.utils import Modality


def get_backend(
    embedding_generator: str,
    distance_metric: str,
    embedding_backend: str,
    embeddings_dir: str,
    metadata_path: str,
    id_column_name: str,
    modality: Modality,
) -> BackendBase:
    backend_config = BackendConfig(
        embeddings_dir=embeddings_dir,
        metadata_path=metadata_path,
        embedding_generator=embedding_generator,
        distance_metric=distance_metric,
        id_column_name=id_column_name,
        modality=modality,
    )

    if embedding_backend == "exact_cpu":
        return NumpyBackend(backend_config)
    elif embedding_backend == "exact_gpu":
        return TorchBackend(backend_config)
    elif embedding_backend == "approx":
        return HNSWBackend(backend_config)
    else:
        raise ValueError(f"Unknown value for embedding_backend, provided: {embedding_backend}")
