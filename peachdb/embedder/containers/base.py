import os
from abc import ABC, abstractmethod
from pathlib import Path

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
)

modal_compute_spec_decorator = lambda stub, image: stub.cls(
    image=image,
    gpu="T4",
    timeout=400,
    secret=modal.Secret.from_dotenv(Path(__file__).parents[3] / ".env"),
    concurrency_limit=500,
)


class EmbeddingModelBase(ABC):
    @abstractmethod
    def _calculate_embeddings(self, texts: list, show_progress_bar: bool):
        pass

    def calculate_embeddings(self, texts: list, ids: list, output_path: str, show_progress_bar: bool = False):
        embeddings = self._calculate_embeddings(texts, show_progress_bar)

        tmp_output_path = "/root/embeddings.parquet"

        df = pd.DataFrame({"ids": ids, "embeddings": embeddings.tolist()})
        table = pa.Table.from_pandas(df)
        pq.write_table(table, tmp_output_path)

        if is_s3_uri(output_path):
            os.system(f"aws s3 cp {tmp_output_path} {output_path}")
            return

        return table
