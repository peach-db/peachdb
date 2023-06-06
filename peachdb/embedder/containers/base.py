import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import boto3  # type: ignore
import modal
import numpy as np
import pandas as pd
import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore
import requests

from peachdb.constants import CACHED_REQUIREMENTS_TXT, GIT_REQUIREMENTS_TXT
from peachdb.embedder.utils import S3File, S3Files, is_s3_uri

# Logic to get a requirements.txt file for the base image when package is on PyPI.
dev_requirements_path = Path(__file__).parents[3] / "requirements.txt"
if os.path.exists(dev_requirements_path):
    requirements_path: Union[Path, str] = dev_requirements_path
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
    .pip_install_from_requirements(str(requirements_path))
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


class EmbeddingModelBase(ABC):
    @abstractmethod
    def _calculate_text_embeddings(self, texts: list, show_progress_bar: bool) -> np.ndarray:
        pass

    @abstractmethod
    def _calculate_audio_embeddings(self, audio_paths: list, show_progress_bar: bool) -> np.ndarray:
        pass

    @abstractmethod
    def _calculate_image_embeddings(self, image_paths: list, show_progress_bar: bool) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def _can_take_text_input(cls) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def _can_take_audio_input(cls) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def _can_take_image_input(cls) -> bool:
        raise NotImplementedError

    def _check_s3_credentials(self):
        try:
            subprocess.check_output(["aws", "s3", "ls", "s3://"], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            raise EnvironmentError(
                "AWS CLI not configured. Please set credentials locally using `aws configure` and try again."
            )

    def calculate_embeddings(
        self,
        ids: list,
        output_path: str,
        texts: Optional[list] = None,
        audio_paths: Optional[list] = None,
        image_paths: Optional[list] = None,
        show_progress_bar: bool = False,
    ) -> Union[None, pa.Table]:
        assert (
            texts is not None or audio_paths is not None or image_paths is not None
        ), "Must provide at least one input."

        if (
            is_s3_uri(output_path)
            # TODO: refactor
            or (any([is_s3_uri(audio_path) for audio_path in audio_paths]) if audio_paths is not None else False)
            or (any([is_s3_uri(image_path) for image_path in image_paths]) if image_paths is not None else False)
        ):
            self._check_s3_credentials()

        embeddings_dict = {}
        embeddings_dict["ids"] = ids

        if texts is not None:
            print("Processing text input...")
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
            print("Processing audio input...")
            if self._can_take_audio_input:
                assert (
                    len(set([is_s3_uri(x) for x in audio_paths])) == 1
                ), "All audio paths must be either local or S3 paths."

                is_s3 = all([is_s3_uri(x) for x in audio_paths])

                if is_s3:
                    print("Downloading audio files from S3... Creating handlers...")
                    audio_files_handler = S3Files(audio_paths)

                    print("Handlers created, downloading...")
                    local_audio_paths = audio_files_handler.download()
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
                    print("Downloading audio files from S3... Creating handlers...")
                    image_file_handlers = S3Files(image_paths)

                    print("Handlers created, downloading...")
                    local_image_paths = image_file_handlers.download()
                else:
                    local_image_paths = image_paths

                image_embeddings = self._calculate_image_embeddings(local_image_paths, show_progress_bar)
                embeddings_dict["image_embeddings"] = image_embeddings.tolist()
            else:
                raise Exception("This model cannot take image input.")

        tmp_output_path = "/root/embeddings.parquet"

        df = pd.DataFrame(embeddings_dict)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, tmp_output_path)

        if is_s3_uri(output_path):
            os.system(f"aws s3 cp {tmp_output_path} {output_path}")
            return None

        return table
