import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

import boto3
import modal
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from peachdb.constants import CACHED_REQUIREMENTS_TXT, GIT_REQUIREMENTS_TXT
from peachdb.embedder.utils import is_s3_uri

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


class EmbeddingModelBase(ABC):
    @abstractmethod
    def _calculate_embeddings(self, texts: list, show_progress_bar: bool):
        pass

    def _check_s3_credentials(self):
        try:
            subprocess.check_output(["aws", "s3", "ls", "s3://"], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            raise EnvironmentError(
                "AWS CLI not configured. Please set credentials locally using `aws configure` and try again."
            )

    def calculate_embeddings(self, texts: list, ids: list, output_path: str, show_progress_bar: bool = False):
        if is_s3_uri(output_path):
            self._check_s3_credentials()

        embeddings = self._calculate_embeddings(texts, show_progress_bar)

        tmp_output_path = "/root/embeddings.parquet"

        df = pd.DataFrame({"ids": ids, "embeddings": embeddings.tolist()})
        table = pa.Table.from_pandas(df)
        pq.write_table(table, tmp_output_path)

        if is_s3_uri(output_path):
            os.system(f"aws s3 cp {tmp_output_path} {output_path}")
            return

        return table
