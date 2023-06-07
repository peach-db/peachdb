from typing import Optional, Union

import modal
import numpy as np
import pyarrow as pa  # type: ignore

from peachdb.embedder.containers.base import EmbeddingModelBase, base_container_image, modal_compute_spec_decorator
from peachdb.embedder.models.audio_clap import AudioClapModel

AUDIOCLAP_BATCH_SIZE = 1024

audioclap_stub = modal.Stub("AudioClap")


def download_model():
    AudioClapModel.download_model()


image = base_container_image.run_function(download_model)


@modal_compute_spec_decorator(stub=audioclap_stub, image=image)
class AudioClapEmbdedder(EmbeddingModelBase):
    def __enter__(self):
        self.model = AudioClapModel()

    def _calculate_text_embeddings(self, texts, show_progress_bar: bool) -> np.ndarray:
        # TODO: fix batch_size
        return self.model.encode_texts(texts, AUDIOCLAP_BATCH_SIZE, show_progress_bar)

    def _calculate_audio_embeddings(self, audio_paths, show_progress_bar: bool) -> np.ndarray:
        # TODO: fix batch_size
        return self.model.encode_audio(audio_paths, AUDIOCLAP_BATCH_SIZE // 8, show_progress_bar)

    def _calculate_image_embeddings(self, image_paths: list, show_progress_bar: bool) -> np.ndarray:
        raise NotImplementedError

    @property
    def _can_take_text_input(cls) -> bool:
        return True

    @property
    def _can_take_audio_input(cls) -> bool:
        return True

    @property
    def _can_take_image_input(cls) -> bool:
        return False

    @modal.method()
    def calculate_embeddings(  # type: ignore
        self,
        ids: list,
        output_path: str,
        texts: Optional[list] = None,
        audio_paths: Optional[list] = None,
        image_paths: Optional[list] = None,
        show_progress_bar: bool = False,
    ) -> Union[None, pa.Table]:
        return super().calculate_embeddings(
            ids=ids,
            output_path=output_path,
            texts=texts,
            audio_paths=audio_paths,
            image_paths=image_paths,
            show_progress_bar=show_progress_bar,
        )


# We have a function here instead of putting it into __main__ so that `modal shell` works
@audioclap_stub.function(image=image)
def test_text(s3_bucket_path: str):
    st = AudioClapEmbdedder()
    embeddings = st.calculate_embeddings.call(
        texts=["hello", "world"],
        ids=[1, 2],
        output_path=s3_bucket_path,
        show_progress_bar=True,
    )


@audioclap_stub.function(image=image)
def test_audio(s3_bucket_path: str):
    st = AudioClapEmbdedder()
    embeddings = st.calculate_embeddings.call(
        audio_paths=[
            "s3://clip-audio-deploy/audioset/---1_cCGK4M.flac",
            "s3://clip-audio-deploy/fma/000002.flac",
            "s3://clip-audio-deploy/freesound/id_=100000.flac",
        ],
        ids=[1, 2, 3],
        output_path=s3_bucket_path,
        show_progress_bar=True,
    )


if __name__ == "__main__":
    # Run as "python sentence_transformer.py s3://<bucket_name>/test_mainfn/"
    import sys

    with audioclap_stub.run():
        # test_text.call(sys.argv[1])
        test_audio.call(sys.argv[1])
