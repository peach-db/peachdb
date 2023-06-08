from typing import List

import numpy as np
import torch
import torchaudio  # type: ignore
import tqdm
from transformers import AutoTokenizer, ClapModel, ClapProcessor  # type: ignore

from peachdb.embedder.models.base import BaseModel


class AudioClapModel(BaseModel):
    def __init__(self) -> None:
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
        self.model = ClapModel.from_pretrained("laion/clap-htsat-fused").eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

    def encode_texts(self, texts, batch_size, show_progress_bar) -> np.ndarray:
        embeddings = []

        for start_index in tqdm.tqdm(range(0, len(texts), batch_size), desc="Batches", disable=not show_progress_bar):
            texts_batch = texts[start_index : start_index + batch_size]
            with torch.no_grad():
                inputs = self.tokenizer(text=texts_batch, return_tensors="pt", padding=True)
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)
                embeddings.append(self.model.get_text_features(**inputs).cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    def encode_audio(self, local_paths, batch_size, show_progress_bar) -> np.ndarray:
        embeddings = []

        for start_index in tqdm.tqdm(
            range(0, len(local_paths), batch_size), desc="Batches", disable=not show_progress_bar
        ):
            batch_local_paths = local_paths[start_index : start_index + batch_size]
            with torch.no_grad():
                input_features = []
                is_longer = []

                for f in batch_local_paths:
                    waveform, sr = torchaudio.load(f)
                    input = self.processor(audios=waveform[0], return_tensors="pt", sampling_rate=sr)

                    input_features.append(input["input_features"])
                    is_longer.append(input["is_longer"])

                input_features = torch.cat(input_features, dim=0).to(self.device)  # type: ignore
                is_longer = torch.cat(is_longer, dim=0).to(self.device)  # type: ignore
                embeddings.append(
                    self.model.get_audio_features(input_features=input_features, is_longer=is_longer).cpu().numpy()
                )

        return np.concatenate(embeddings, axis=0)

    def encode_image(self, local_paths, batch_size, show_progress_bar) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def download_model():
        ClapProcessor.from_pretrained("laion/clap-htsat-fused")
        ClapModel.from_pretrained("laion/clap-htsat-fused")
        AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")
