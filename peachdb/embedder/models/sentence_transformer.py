import torch
from sentence_transformers import SentenceTransformer

from peachdb.embedder.models.base import BaseModel, Modality, SingleModalityDataset

MODEL_NAME = "all-MiniLM-L12-v2"


class SentenceTransformerModel(BaseModel):
    def __init__(self) -> None:
        self.model = SentenceTransformer(MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu")

    def get_preprocessor(self, modality: Modality):
        if isinstance(modality, Modality.TEXT):
            # Not needed as SentenceTransformerModel handles preprocessing internally
            return lambda x: x
        else:
            raise Exception(f"Unsupported modality: {modality}")

    def _encode_batch(self, batch_data, dataset_modality, preprocessor):
        # Not needed as SentenceTransformerModel handles batching internally. `encode_single_modality` uses it
        # but that's overridden below.
        pass

    def encode_single_modality(self, dataset: SingleModalityDataset, batch_size, show_progress_bar):
        # Override base class method as sentence_transformers does a bunch of preprocessing internally

        if dataset.modality != Modality.TEXT:
            raise Exception(f"Unsupported modality: {dataset.modality}")

        return self.model.encode(dataset.data, batch_size=batch_size, show_progress_bar=show_progress_bar)

    def encode_multiple_modalities(
        self,
        datasets: list[SingleModalityDataset],
        batch_size,
        show_progress_bar,
    ):
        raise NotImplementedError("Sentence Transformers do not support multiple modalities")

    @staticmethod
    def download_model():
        SentenceTransformer(MODEL_NAME, device="cpu")
