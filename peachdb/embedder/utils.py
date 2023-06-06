import abc
import concurrent.futures
import datetime
import logging
import os
import subprocess
import tempfile
import uuid
from enum import Enum
from functools import cache, wraps
from types import SimpleNamespace
from typing import List, Optional, Type, Union

import tqdm  # type: ignore

logger = logging.getLogger(__name__)


class Modality(Enum):
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"


def handle_s3_download_error(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            message = f"Failed to download {args[0]} from S3. Error: {str(e)}"
            logger.exception(message)
            raise FileNotFoundError(message)

    return wrapper


@cache
def _verify_aws_cli_installed() -> None:
    try:
        subprocess.check_output(["aws", "--version"], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        raise EnvironmentError("AWS CLI not installed. Please install it and try again.")


class S3Entity(metaclass=abc.ABCMeta):
    """Abstract base class for S3 Entities"""

    def __init__(self, s3_path: str):
        _verify_aws_cli_installed()
        self.s3_path = s3_path
        self.temp_resource: Optional[Union[tempfile._TemporaryFileWrapper, tempfile.TemporaryDirectory]] = None

    @abc.abstractmethod
    def download(self):
        pass

    @abc.abstractmethod
    def cleanup(self):
        pass

    def __enter__(self) -> str:
        self.download()

        assert self.temp_resource is not None
        return self.temp_resource.name

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    @handle_s3_download_error
    def _download_from_s3(self, command: str) -> None:
        assert self.temp_resource is not None

        with subprocess.Popen(
            ["aws", "s3", command, self.s3_path, self.temp_resource.name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        ) as download_process:
            assert download_process.stdout is not None
            for line in download_process.stdout:
                print("\r" + line.strip().ljust(120), end="", flush=True)
            print()


class S3Files(S3Entity):
    def __init__(self, s3_paths: list[str]):
        _verify_aws_cli_installed()
        self.s3_paths = s3_paths
        self.temp_resources: Optional[List[tempfile._TemporaryFileWrapper]] = None

    @handle_s3_download_error
    def copy_file(self, path, resource_name) -> None:
        command = ["aws", "s3", "cp", path, resource_name.name]

        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        ) as download_process:
            assert download_process.stdout is not None
            for line in download_process.stdout:
                pass

    def _download_from_s3(self) -> None:
        assert self.temp_resources is not None

        print("CPU count: ", os.cpu_count())

        futures = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.s3_paths)) as executor:
            for path, resource in tqdm.tqdm(
                zip(self.s3_paths, self.temp_resources), total=len(self.s3_paths), desc="Scheduling download threads"
            ):
                futures.append(executor.submit(self.copy_file, path, resource))

            for future in tqdm.tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Waiting for threads to finish",
            ):
                future.result()

    def cleanup(self):
        [x.cleanup() for x in self.temp_resources]

    def download(self) -> list[str]:
        self.temp_resources = [tempfile.NamedTemporaryFile(delete=True) for _ in tqdm.tqdm(self.s3_paths)]
        self._download_from_s3()
        return [x.name for x in self.temp_resources]


class S3File(S3Entity):
    """Represents an S3 File"""

    def download(self) -> str:
        self.temp_resource = tempfile.NamedTemporaryFile(delete=True)
        self._download_from_s3("cp")
        return self.temp_resource.name

    def cleanup(self):
        self.temp_resource.close()


class S3Folder(S3Entity):
    """Represents an S3 Folder"""

    def download(self) -> str:
        self.temp_resource = tempfile.TemporaryDirectory()
        self._download_from_s3("sync")
        return self.temp_resource.name

    def cleanup(self):
        self.temp_resource.cleanup()


def create_unique_id() -> str:
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    return now_str + "_" + str(uuid.uuid4())


def create_unique_parquet_name() -> str:
    return f"{create_unique_id()}.parquet"


def is_s3_uri(path: str) -> bool:
    return "s3://" in path
