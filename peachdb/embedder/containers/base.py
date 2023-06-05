import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import boto3
import modal
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from peachdb.constants import CACHED_REQUIREMENTS_TXT, GIT_REQUIREMENTS_TXT
from peachdb.embedder.utils import S3File, is_s3_uri

# Logic to get a requirements.txt file for the base image when package is on PyPI.
dev_requirements_path = Path(__file__).parents[3] / "requirements.txt"
if os.path.exists(dev_requirements_path):
    requirements_path = dev_requirements_path
else:
    response = requests.get(GIT_REQUIREMENTS_TXT)

    response.raise_for_status()

    with open(CACHED_REQUIREMENTS_TXT, "w") as f:
        f.write(response.text)

    requirements_path = CACHED_REQUIREMENTS_TXT

# Grab AWS credentials from ~/.aws/credentials using boto3.
# This code is written this way as it ends up getting executed inside the container creation process as well,
# and so ends up with empty credentials. Doing it this way means empty credentials get set in the container creation process,
# but the actual model serving process will have the correct credentials.
# TODO: fix scope for bad error handling here. really want to error if these don't exist locally!
# But we don't have stub here to run `is_inside`.
secrets = []
_aws_boto_session = boto3.Session()
if _aws_boto_session != None:
    _aws_credentials = _aws_boto_session.get_credentials()
    if _aws_credentials != None:
        secrets = [
            modal.Secret.from_dict(
                {"AWS_ACCESS_KEY_ID": _aws_credentials.access_key, "AWS_SECRET_ACCESS_KEY": _aws_credentials.secret_key}
            ),
        ]

# Requirements for the base image of models we want to serve.
# We don't add the requirements.txt here as that contains requirements across ALL our models.
base_container_image = (
    modal.Image.debian_slim()
    .apt_install("curl", "zip", "git")
    .run_commands(
        "curl https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o awscliv2.zip",
        "unzip awscliv2.zip",
        "./aws/install",
        "rm -rf awscliv2.zip aws",
    )
    .pip_install_from_requirements(requirements_path)
    # Container creation flow from inside a pypi package doesn't pick up it's own files.
    .pip_install("git+https://github.com/peach-db/peachdb")
)

modal_compute_spec_decorator = lambda stub, image: stub.cls(
    image=image,
    gpu="T4",
    timeout=400,
    secrets=secrets,
    concurrency_limit=500,
)

# TODO: refactor.
import numpy as np
from embedder.models.base import Modality, SingleModalityDataset


class SingleModalityRemoteDataset:
    def __init__(self, modality: Modality, ids: np.ndarray, data: list):
        self.modality = modality
        self.ids = ids
        self.data = data

        if self._is_data_on_s3():
            self._check_s3_credentials()
            # TODO: now this download might not get executed on Modal!
            self._download_data_locally()

    def _is_data_on_s3(self):
        is_s3_uri_paths = [is_s3_uri(x) for x in self.paths]

        if len(set(is_s3_uri_paths)) != 1:
            raise ValueError("All paths must be either local or S3 paths.")

        return is_s3_uri_paths[0]

    def _download_data_locally(self):
        self._remote_file_handlers = [S3File(path) for path in self.data]
        self._remote_data = self.data
        self.data = [
            file_handler.download() for file_handler in self._remote_file_handlers
        ]  # TODO: We should parallelise downloading this data!


class EmbeddingModelBase(ABC):
    @abstractmethod
    def _calculate_embeddings(self, dataset: SingleModalityDataset, show_progress_bar: bool):
        pass

    def _check_s3_credentials(self):
        try:
            subprocess.check_output(["aws", "s3", "ls", "s3://"], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            raise EnvironmentError(
                "AWS CLI not configured. Please set credentials locally using `aws configure` and try again."
            )

    @abstractmethod
    def _calculate_embeddings(self, datasets: list[SingleModalityDataset], show_progress_bar: bool):
        pass

    def calculate_embeddings(
        self,
        datasets: list[SingleModalityRemoteDataset],
        output_path: str,
        show_progress_bar: bool = False,
    ):
        if is_s3_uri(output_path):
            self._check_s3_credentials()

        embeddings_dict = {}
        # TODO: add ids back in.
        # embeddings_dict["ids"] = ids

        local_datasets = [dataset.get_local_dataset() for dataset in datasets]
        embeddings_dict.update(self._calculate_embeddings(local_datasets, show_progress_bar))

        tmp_output_path = "/root/embeddings.parquet"

        df = pd.DataFrame(embeddings_dict)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, tmp_output_path)

        if is_s3_uri(output_path):
            os.system(f"aws s3 cp {tmp_output_path} {output_path}")
            return

        return table


"""

        if texts is not None:
            if self._can_take_text_input:
                text_embeddings = self._calculate_text_embeddings(texts, show_progress_bar)
                # TODO: update wherever this variable is used upstream
                embeddings_dict["text_embeddings"] = text_embeddings.tolist()
            else:
                raise Exception("This model cannot take text input.")

        # TODO: refactor below two if statements.
        # TODO: think about if we want to error if a modality is not supported by a model, or just
        # error, and then let things continue.
        if audio_paths is not None:
            if self._can_take_audio_input:
                assert (
                    len(set([is_s3_uri(x) for x in audio_paths])) == 1
                ), "All audio paths must be either local or S3 paths."

                is_s3 = all([is_s3_uri(x) for x in audio_paths])

                if is_s3:
                    
                else:
                    local_audio_paths = audio_paths

                audio_embeddings = self._calculate_audio_embeddings(local_audio_paths, show_progress_bar)
                embeddings_dict["audio_embeddings"] = audio_embeddings.tolist()
            else:
                raise Exception("This model cannot take audio input.")

        if image_paths is not None:
            if self._can_take_image_input:
                assert (
                    len(set([is_s3_uri(x) for x in image_paths])) == 1
                ), "All image paths must be either local or S3 paths."

                is_s3 = all([is_s3_uri(x) for x in image_paths])

                if is_s3:
                    image_file_handlers = [S3File(path) for path in image_paths]
                    local_image_paths = [
                        file_handler.download() for file_handler in image_file_handlers
                    ]  # TODO: We should parallelise downloading this data!
                else:
                    local_image_paths = image_paths

                image_embeddings = self._calculate_image_embeddings(local_image_paths, show_progress_bar)
                embeddings_dict["image_embeddings"] = image_embeddings.tolist()
            else:
                raise Exception("This model cannot take image input.")


"""
