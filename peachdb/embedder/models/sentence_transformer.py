import torch
from sentence_transformers import SentenceTransformer

from peachdb.embedder.models.base import BaseModel

MODEL_NAME = "all-MiniLM-L12-v2"


class SentenceTransformerModel(BaseModel):
    def __init__(self) -> None:
        self.model = SentenceTransformer(MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu")

    def encode(self, texts, batch_size, show_progress_bar):
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress_bar)

    @staticmethod
    def download_model():
        SentenceTransformer(MODEL_NAME, device="cpu")
