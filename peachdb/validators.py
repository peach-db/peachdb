import os
from typing import Optional

import numpy as np
import pandas as pd

from peachdb.embedder.utils import S3File, is_s3_uri


def validate_embedding_generator(engine: str):
    supported_engines = ["sentence_transformer_L12", "imagebind"]
    assert (
        engine in supported_engines
    ), f"Unsupported embedding generator. Currently supported engines are: {', '.join(supported_engines)}"


def validate_distance_metric(metric: str):
    supported_metrics = ["l2", "cosine"]
    assert (
        metric in supported_metrics
    ), f"Unsupported distance metric. The metric should be one of the following: {', '.join(supported_metrics)}"


def validate_embedding_backend(backend: str):
    supported_backends = ["exact_cpu", "exact_gpu", "approx"]
    assert (
        backend in supported_backends
    ), f"Unsupported embedding backend. The backend should be one of the following: {', '.join(supported_backends)}"


def validate_csv_path(csv_path: str):
    assert csv_path, "csv_path parameter is missing. Please provide a valid local file path or an S3 URI."

    # TODO: in case of S3 URI, check if the URI exists
    if not is_s3_uri(csv_path):
        assert os.path.exists(
            csv_path
        ), f"The provided csv_path '{csv_path}' does not exist. Please ensure that the path is either a valid local file path or an S3 URI (e.g., s3://path/to/csv)."


def validate_columns(column_to_embed: str, id_column_name: str, csv_path: str):
    assert column_to_embed
    assert id_column_name

    def _check(data: pd.DataFrame):
        assert column_to_embed in data.columns, f"column_to_embed parameter is missing in {data.columns}"
        assert id_column_name in data.columns, f"id_column_name parameter is missing in {data.columns}"

        try:
            ids = np.array(data[id_column_name].values.tolist()).astype("int64")
        except:
            print("[red]Only INTEGER datatype is supported for id column right now[/]")

    if is_s3_uri(csv_path):
        with S3File(csv_path) as downloaded_csv:
            data = pd.read_csv(downloaded_csv)
            _check(data)
    else:
        data = pd.read_csv(csv_path)
        _check(data)
