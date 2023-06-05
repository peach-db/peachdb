import numpy as np
import torch
import tqdm  # type: ignore
from imagebind.data import (  # type: ignore
    load_and_transform_audio_data,
    load_and_transform_text,
    load_and_transform_vision_data,
)
from imagebind.models import imagebind_model  # type: ignore
from imagebind.models.imagebind_model import ModalityType  # type: ignore

from peachdb.embedder.models.base import BaseModel


class ImageBindModel(BaseModel):
    def __init__(self) -> None:
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"  # TODO: can we factor to base model?

        self.model = imagebind_model.imagebind_huge(pretrained=True).eval().to(self.device)

    # TODO: we want all the encodings in one function so that we can get the embeddings in one go!
    # Otherwise we are wasting compute. Refactor given this.
    # HAVE A SINGLE ENCODE FUNCTION!

    def encode_texts(self, texts, batch_size, show_progress_bar) -> np.ndarray:
        embeddings = []
        for start_index in tqdm.tqdm(range(0, len(texts), batch_size), desc="Batches", disable=not show_progress_bar):
            texts_batch = texts[start_index : start_index + batch_size]
            inputs_batch = {ModalityType.TEXT: load_and_transform_text(texts_batch, self.device)}

            with torch.no_grad():
                embeddings.append(self.model(inputs_batch)[ModalityType.TEXT].cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    def encode_audio(self, local_paths, batch_size, show_progress_bar) -> np.ndarray:
        embeddings = []
        for start_index in tqdm.tqdm(
            range(0, len(local_paths), batch_size), desc="Batches", disable=not show_progress_bar
        ):
            batch_local_paths = local_paths[start_index : start_index + batch_size]
            batched_inputs = {ModalityType.AUDIO: load_and_transform_audio_data(batch_local_paths, self.device)}

            with torch.no_grad():
                embeddings.append(self.model(batched_inputs)[ModalityType.AUDIO].cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    def encode_image(self, local_paths, batch_size, show_progress_bar) -> np.ndarray:
        embeddings = []
        for start_index in tqdm.tqdm(
            range(0, len(local_paths), batch_size), desc="Batches", disable=not show_progress_bar
        ):
            batch_local_paths = local_paths[start_index : start_index + batch_size]
            batched_inputs = {ModalityType.VISION: load_and_transform_vision_data(batch_local_paths, self.device)}

            with torch.no_grad():
                embeddings.append(self.model(batched_inputs)[ModalityType.VISION].cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    @staticmethod
    def download_model():
        imagebind_model.imagebind_huge(pretrained=True)
