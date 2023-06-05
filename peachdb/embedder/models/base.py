import abc
from types import SimpleNamespace

import numpy as np
import tqdm

Modality = SimpleNamespace(
    TEXT="text",
    AUDIO="audio",
    IMAGE="image",
)


class SingleModalityDataset:
    modality: Modality
    ids: list[np.ndarray]
    data: list[np.ndarray]

    def get_batch(self, start_idx, end_idx):
        return self.data[start_idx:end_idx]  # , self.ids[start_idx:end_idx]


class BaseModel(abc.ABC):
    @abc.abstractmethod
    def __init__(self) -> None:
        pass

    def encode_single_modality(
        self, dataset: SingleModalityDataset, batch_size: int, show_progress_bar: bool
    ) -> np.ndarray:
        embeddings = []

        modality_preprocessor = self.get_preprocessor(dataset.modality)

        for start_index in tqdm.tqdm(
            range(0, len(dataset.data), batch_size),
            desc="Batches",
            disable=not show_progress_bar,
        ):
            batch_data, _ = dataset.get_batch(start_index, start_index + batch_size)
            batch_embeddings = self._encode_batch(batch_data, dataset.modality, modality_preprocessor)

            embeddings.append(batch_embeddings)

        return np.concatenate(embeddings, axis=0)

    def encode_multiple_modalities(
        self, datasets: list[SingleModalityDataset], batch_size: int, show_progress_bar: bool
    ) -> tuple[np.ndarray, ...]:
        # TODO: how do we use batch_size? Maybe dict of Modality : batch_size? (can be done later!)
        modalities = [dataset.modality for dataset in datasets]
        if len(set(modalities)) != len(modalities):
            raise Exception("Duplicate modalities in datasets. This is unexpected.")

        return {
            dataset.modality: self.encode_single_modality(dataset, batch_size, show_progress_bar)
            for dataset in datasets
        }

    @abc.abstractmethod
    def get_preprocessor(self, modality):
        pass

    @abc.abstractmethod
    def _encode_batch(self, batch_data, dataset_modality, preprocessor):
        pass

    @staticmethod
    @abc.abstractmethod
    def download_model():
        pass
