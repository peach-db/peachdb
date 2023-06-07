from typing import List

import numpy as np
import torch
import torchaudio  # type: ignore
from transformers import AutoTokenizer, ClapModel, ClapProcessor  # type: ignore

from peachdb.embedder.models.base import BaseModel


class AudioClapModel(BaseModel):
    def __init__(self) -> None:
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # TODO: should we use fused or not?
        self.processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
        self.model = ClapModel.from_pretrained("laion/clap-htsat-fused").eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

    def encode_texts(self, texts, batch_size, show_progress_bar) -> np.ndarray:
        # assert batch_size is None and show_progress_bar is None, "not yet implemented."
        # TODO: add batching logic
        # TODO: abstract away batching logic.

        with torch.no_grad():
            inputs = self.tokenizer(text=texts, return_tensors="pt", padding=True)
            for k, v in inputs.items():
                inputs[k] = v.to(self.device)
            outputs = self.model.get_text_features(**inputs)

        return outputs.cpu().numpy()

    def encode_audio(self, local_paths, batch_size, show_progress_bar) -> np.ndarray:
        # assert batch_size is None and show_progress_bar is None, "not yet implemented."
        # TODO: add batching logic
        # TODO: abstract away batching logic.

        with torch.no_grad():
            input_features = []
            is_longer = []

            for f in local_paths:
                waveform, sr = torchaudio.load(f)
                input = self.processor(audios=waveform[0], return_tensors="pt", sampling_rate=sr)

                input_features.append(input["input_features"])
                is_longer.append(input["is_longer"])

            input_features = torch.cat(input_features, dim=0).to(self.device)  # type: ignore
            is_longer = torch.cat(is_longer, dim=0).to(self.device)  # type: ignore
            audio_embed = self.model.get_audio_features(input_features=input_features, is_longer=is_longer)
            audio_embed = audio_embed.cpu().numpy()

        return audio_embed

    def encode_image(self, local_paths, batch_size, show_progress_bar) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def download_model():
        ClapProcessor.from_pretrained("laion/clap-htsat-fused")
        ClapModel.from_pretrained("laion/clap-htsat-fused")
        AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")
