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

# Download data -- can be abstracted

audio_file_handlers = [S3File(row["audios"]) for index, row in df.iterrows()]
audio_paths = [
    file_handler.download() for file_handler in audio_file_handlers
]  # We should parallelise downloading this data!

# Common initialisation -- can be abstracted!
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Define the model

# ## CLAP

# ### Initialisation (also downloads model!)
# # TODO: should we use fused or unfused models?
# model = ClapModel.from_pretrained("laion/clap-htsat-fused").eval().to(device)
# processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")

# # TODO: make this work with peachdb abstractions
# # TODO: add CLAP to container

# torch_audios = [torchaudio.load(x) for x in audio_paths]
# inputs = [processor(audios=aud_2d[0], return_tensors="pt", sampling_rate=sr) for aud_2d, sr in torch_audios]
# # [x["input_features"].shape for x in inputs]
# # [x["is_longer"] for x in inputs]
# input_features = torch.cat([x["input_features"] for x in inputs], dim=0).to(device)
# is_longer = torch.cat([x["is_longer"] for x in inputs], dim=0).to(device)

# with torch.no_grad():
#     audio_embed = model.get_audio_features(input_features=input_features, is_longer=is_longer)

# print(audio_embed.cpu().shape)

## ImageBind

### Initialisation (also downloads model!)
import imagebind.data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

model = imagebind_model.imagebind_huge(pretrained=True).eval().to(device)

### Inference
inputs = {ModalityType.AUDIO: imagebind.data.load_and_transform_audio_data(audio_paths, device)}

with torch.no_grad():
    audio_embed = model(inputs)  # get a dict with [3, 1024] as output.

print(audio_embed[ModalityType.AUDIO].cpu().shape)
