import numpy as np
import torch
import tqdm
from imagebind.data import load_and_transform_text
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

from peachdb.embedder.models.base import BaseModel


class ImageBindModel(BaseModel):
    def __init__(self) -> None:
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model = self.model.eval()
        self.model = self.model.to(self.device)

    def encode(self, texts, batch_size, show_progress_bar):
        # TODO: enable ability to encode audio and video?
        embeddings = []
        for start_index in tqdm.tqdm(range(0, len(texts), batch_size), desc="Batches", disable=not show_progress_bar):
            texts_batch = texts[start_index : start_index + batch_size]
            inputs_batch = {ModalityType.TEXT: load_and_transform_text(texts_batch, self.device)}

            with torch.no_grad():
                embeddings.append(self.model(inputs_batch)[ModalityType.TEXT].cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    @staticmethod
    def download_model():
        imagebind_model.imagebind_huge(pretrained=True)
