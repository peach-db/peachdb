import numpy as np
import torch
import tqdm
from imagebind.data import load_and_transform_audio_data, load_and_transform_text
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

from peachdb.embedder.models.base import BaseModel


class ImageBindModel(BaseModel):
    def __init__(self) -> None:
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"  # TODO: can we factor to base model?

        self.model = imagebind_model.imagebind_huge(pretrained=True).eval().to(self.device)

    # TODO: we want all the encodings in one function so that we can get the embeddings in one go!
    # Otherwise we are wasting compute. Refactor given this.
    # HAVE A SINGLE ENCODE FUNCTION!

    def encode_text(self, texts, batch_size, show_progress_bar):
        embeddings = []
        for start_index in tqdm.tqdm(range(0, len(texts), batch_size), desc="Batches", disable=not show_progress_bar):
            texts_batch = texts[start_index : start_index + batch_size]
            inputs_batch = {ModalityType.TEXT: load_and_transform_text(texts_batch, self.device)}

            with torch.no_grad():
                embeddings.append(self.model(inputs_batch)[ModalityType.TEXT].cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    def encode_audio(self, local_paths, batch_size, show_progress_bar):
        # TODO: add batching & progress bar

        inputs = {ModalityType.AUDIO: load_and_transform_audio_data(local_paths, self.device)}

        with torch.no_grad():
            audio_embed = self.model(inputs)[ModalityType.AUDIO].cpu().numpy()

        return audio_embed

    def encode_image(self, local_paths, batch_size, show_progress_bar):
        raise NotImplementedError

    @staticmethod
    def download_model():
        imagebind_model.imagebind_huge(pretrained=True)
