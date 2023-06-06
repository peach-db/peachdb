import abc


class BaseModel(abc.ABC):
    @abc.abstractmethod
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def encode_texts(self, texts, batch_size, show_progress_bar):
        pass

    @abc.abstractmethod
    def encode_audio(self, local_paths, batch_size, show_progress_bar):
        pass

    @abc.abstractmethod
    def encode_image(self, local_paths, batch_size, show_progress_bar):
        pass

    @staticmethod
    @abc.abstractmethod
    def download_model():
        pass
