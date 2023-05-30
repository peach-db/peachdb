import abc


class BaseModel(abc.ABC):
    @abc.abstractmethod
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def encode(self, texts, batch_size, show_progress_bar):
        pass

    @staticmethod
    @abc.abstractmethod
    def download_model():
        pass
