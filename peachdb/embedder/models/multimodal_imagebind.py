import numpy as np
import torch
import tqdm
from imagebind.data import load_and_transform_audio_data, load_and_transform_text, load_and_transform_vision_data
from imagebind.models import imagebind_model

from peachdb.embedder.models.base import BaseModel, Modality, SingleModalityDataset


class ImageBindModel(BaseModel):
    def __init__(self) -> None:
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"  # TODO: can we factor to base model?

        self.model = imagebind_model.imagebind_huge(pretrained=True).eval().to(self.device)

    @staticmethod
    def peachdb_to_imagedb_modality_type(modality: Modality):
        return {
            Modality.TEXT: imagebind_model.ModalityType.TEXT,
            Modality.AUDIO: imagebind_model.ModalityType.AUDIO,
            Modality.IMAGE: imagebind_model.ModalityType.IMAGE,
        }

    def get_preprocessor(self, modality: Modality):
        if isinstance(modality, Modality.TEXT):
            return lambda data: load_and_transform_text(data, self.device)
        elif isinstance(modality, Modality.AUDIO):
            return lambda data: load_and_transform_audio_data(data, self.device)
        elif isinstance(modality, Modality.IMAGE):
            return lambda data: load_and_transform_vision_data(data, self.device)
        else:
            raise Exception(f"Unsupported modality: {modality}")

    def _encode_batch(self, batch_data, dataset_modality, preprocessor):
        inputs_batch = {}
        imagebind_modality = self.peachdb_to_imagedb_modality_type(dataset_modality)
        inputs_batch[imagebind_modality] = preprocessor(batch_data)

        with torch.no_grad():
            batch_embeddings = self.model(inputs_batch)

            return batch_embeddings[imagebind_modality].cpu().numpy()

    @staticmethod
    def download_model():
        imagebind_model.imagebind_huge(pretrained=True)
