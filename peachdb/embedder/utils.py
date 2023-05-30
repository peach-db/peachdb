import abc
import datetime
import logging
import subprocess
import tempfile
import uuid
from functools import wraps

logger = logging.getLogger(__name__)


def handle_s3_download_error(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except subprocess.CalledProcessError as e:
            message = f"Failed to download {args[0]} from S3. Error: {str(e)}"
            logger.exception(message)
            raise FileNotFoundError(message)

    return wrapper


class S3Entity(metaclass=abc.ABCMeta):
    """Abstract base class for S3 Entities"""

    def __init__(self, s3_path: str):
        self._verify_aws_cli_installed()
        self.s3_path = s3_path
        self.temp_resource = None

    @abc.abstractmethod
    def download(self):
        pass

    @abc.abstractmethod
    def cleanup(self):
        pass

    def __enter__(self):
        self.download()
        return self.temp_resource.name

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    @staticmethod
    def _verify_aws_cli_installed() -> None:
        try:
            subprocess.check_output(["aws", "--version"], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            raise EnvironmentError("AWS CLI not installed. Please install it and try again.")

    @handle_s3_download_error
    def _download_from_s3(self, command: str) -> None:
        with subprocess.Popen(
            ["aws", "s3", command, self.s3_path, self.temp_resource.name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        ) as download_process:
            for line in download_process.stdout:
                print("\r" + line.strip().ljust(120), end="", flush=True)
            print()


class S3File(S3Entity):
    """Represents an S3 File"""

    def download(self):
        self.temp_resource = tempfile.NamedTemporaryFile(delete=True)
        self._download_from_s3("cp")
        return self.temp_resource.name

    def cleanup(self):
        self.temp_resource.close()


class S3Folder(S3Entity):
    """Represents an S3 Folder"""

    def download(self):
        self.temp_resource = tempfile.TemporaryDirectory()
        self._download_from_s3("sync")
        return self.temp_resource.name

    def cleanup(self):
        self.temp_resource.cleanup()


def create_unique_id() -> str:
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    return now_str + "_" + str(uuid.uuid4())


def create_unique_parquet_name():
    return f"{create_unique_id()}.parquet"


def is_s3_uri(path: str):
    return "s3://" in path
