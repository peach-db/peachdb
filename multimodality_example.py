import pandas as pd
import torch
import torchaudio
from transformers import ClapModel, ClapProcessor

from peachdb.embedder.utils import S3File

# Define the data
df = pd.DataFrame(
    {
        "audios": [
            "s3://clip-audio-deploy/audioset/---1_cCGK4M.flac",
            "s3://clip-audio-deploy/fma/000002.flac",
            "s3://clip-audio-deploy/freesound/id_=100000.flac",
        ],
        "ids": [1, 2, 3],
    }
)

print(df)

# Define the model
# TODO: should we use fused or unfused models?
model = ClapModel.from_pretrained("laion/clap-htsat-fused")
processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")

# TODO: make this work with peachdb abstractions
# TODO: run the same with ImageBind?
# TODO: add CLAP to container

waveforms = None
for index, row in df.iterrows():
    with S3File(row["audios"]) as s3_file:
        waveform, sr = torchaudio.load(s3_file)
        if waveforms is None:
            waveforms = waveform
        else:
            waveforms = torch.cat((waveforms, waveform), dim=0)
            print(waveforms.shape)

        inputs = processor(audios=waveform[0], return_tensors="pt", sampling_rate=sr)
        audio_embed = model.get_audio_features(**inputs)

        print(audio_embed.shape)
