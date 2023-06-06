from typing import Optional, Union

import modal
import numpy as np
import pyarrow as pa  # type: ignore

from peachdb.embedder.containers.base import EmbeddingModelBase, base_container_image, modal_compute_spec_decorator
from peachdb.embedder.models.multimodal_imagebind import ImageBindModel
from peachdb.embedder.utils import S3File, is_s3_uri

IMAGEBIND_BATCH_SIZE = 1024

imagebind_stub = modal.Stub("ImageBind")


def download_model():
    ImageBindModel.download_model()


image = base_container_image.run_function(download_model)


@modal_compute_spec_decorator(stub=imagebind_stub, image=image)
class ImageBindEmbdedder(EmbeddingModelBase):
    def __enter__(self):
        self.model = ImageBindModel()

    def _calculate_text_embeddings(self, texts, show_progress_bar: bool) -> np.ndarray:
        return self.model.encode_texts(texts, IMAGEBIND_BATCH_SIZE, show_progress_bar)

    def _calculate_audio_embeddings(self, audio_paths, show_progress_bar: bool) -> np.ndarray:
        # TODO: add handling of different batch sizes to EmbeddingProcessor. (#multi-modality)
        return self.model.encode_audio(audio_paths, IMAGEBIND_BATCH_SIZE // 8, show_progress_bar)

    def _calculate_image_embeddings(self, image_paths: list, show_progress_bar: bool) -> np.ndarray:
        # TODO: add handling of different batch sizes to EmbeddingProcessor. (#multi-modality)
        return self.model.encode_image(image_paths, IMAGEBIND_BATCH_SIZE // 8, show_progress_bar)

    @property
    def _can_take_text_input(cls) -> bool:
        return True

    @property
    def _can_take_audio_input(cls) -> bool:
        return True

    @property
    def _can_take_image_input(cls) -> bool:
        return True

    # We need to rewrite this function in all the inherited class so we can use the @modal method decorator.
    # TODO: check if above statement is true / if we can factor this out.
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


### Test functions ###
# We have a function here instead of putting it into __main__ so that `modal shell` works
@imagebind_stub.function(image=image)
def test_texts(s3_bucket_path: str):
    ib = ImageBindEmbdedder()
    embeddings = ib.calculate_embeddings.call(
        texts=["hello", "world"],
        ids=[1, 2],
        output_path=s3_bucket_path,
        show_progress_bar=True,
    )


@imagebind_stub.function(image=image)
def test_audio(s3_bucket_path: str):
    ib = ImageBindEmbdedder()
    embeddings = ib.calculate_embeddings.call(
        audio_paths=["s3://clip-audio-deploy/audioset/---1_cCGK4M.flac"] * 2,
        ids=list(range(2)),
        output_path=s3_bucket_path,
        show_progress_bar=True,
    )


@imagebind_stub.function(image=image)
def test_texts_audio(s3_bucket_path: str):
    ib = ImageBindEmbdedder()
    embeddings = ib.calculate_embeddings.call(
        texts=["hello", "world"],
        audio_paths=["s3://clip-audio-deploy/audioset/---1_cCGK4M.flac"] * 2,
        ids=list(range(2)),
        output_path=s3_bucket_path,
        show_progress_bar=True,
    )


@imagebind_stub.function(image=image)
def test_texts_audio_batched(s3_bucket_path: str):
    ib = ImageBindEmbdedder()
    embeddings = ib.calculate_embeddings.call(
        texts=["hello"] * 1025,
        audio_paths=["s3://clip-audio-deploy/audioset/---1_cCGK4M.flac"] * 1025,
        ids=list(range(1025)),
        output_path=s3_bucket_path,
        show_progress_bar=True,
    )


@imagebind_stub.function(image=image)
def test_audio_batched(s3_bucket_path: str):
    ib = ImageBindEmbdedder()
    embeddings = ib.calculate_embeddings.call(
        # texts=["hello"] * 1025,
        audio_paths=["s3://clip-audio-deploy/audioset/---1_cCGK4M.flac"] * 10000,
        ids=list(range(1025)),
        output_path=s3_bucket_path,
        show_progress_bar=True,
    )


@imagebind_stub.function(image=image)
def test_image(s3_bucket_path: str):
    ib = ImageBindEmbdedder()
    NUM_IMAGES = 1025
    embeddings = ib.calculate_embeddings.call(
        image_paths=["s3://metavoice-vector-db/bird_image.jpg"] * NUM_IMAGES,
        ids=list(range(NUM_IMAGES)),
        output_path=s3_bucket_path,
        show_progress_bar=True,
    )


if __name__ == "__main__":
    # Run as "python multimodal_imagebind.py s3://<bucket_name>/test_mainfn/"
    import sys

    with imagebind_stub.run():
        # test_texts.call(sys.argv[1])
        # test_audio.call(sys.argv[1])
        # test_texts_audio.call(sys.argv[1])
        # test_texts_audio_batched.call(sys.argv[1])
        # test_audio_batched.call(sys.argv[1])
        test_image.call(sys.argv[1])
