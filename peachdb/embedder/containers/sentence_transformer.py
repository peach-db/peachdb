from typing import Optional, Union

import modal
import pyarrow as pa  # type: ignore

from peachdb.embedder.containers.base import EmbeddingModelBase, base_container_image, modal_compute_spec_decorator
from peachdb.embedder.models.sentence_transformer import SentenceTransformerModel

SENTENCE_TRANSFORMER_BATCH_SIZE = 64

sbert_stub = modal.Stub("SBERT")


def download_model():
    SentenceTransformerModel.download_model()


image = base_container_image.run_function(download_model)


@modal_compute_spec_decorator(stub=sbert_stub, image=image)
class SentenceTransformerEmbedder(EmbeddingModelBase):
    def __enter__(self):
        self.model = SentenceTransformerModel()

    def _calculate_text_embeddings(self, texts, show_progress_bar: bool):
        return self.model.encode_texts(
            texts,
            batch_size=SENTENCE_TRANSFORMER_BATCH_SIZE,
            show_progress_bar=show_progress_bar,
        )

    # TODO: are defining these functions necessary?
    def _calculate_audio_embeddings(self, audio_paths, show_progress_bar: bool):
        raise NotImplementedError

    def _calculate_image_embeddings(self, image_paths: list, show_progress_bar: bool):
        raise NotImplementedError

    @property
    def _can_take_text_input(cls):
        return True

    @property
    def _can_take_audio_input(cls):
        return False

    @property
    def _can_take_image_input(cls):
        # return True - once implemented!
        return False

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


# We have a function here instead of putting it into __main__ so that `modal shell` works
@sbert_stub.function(image=image)
def test(s3_bucket_path: str):
    st = SentenceTransformerEmbedder()
    embeddings = st.calculate_embeddings.call(
        texts=["hello", "world"],
        ids=[1, 2],
        output_path=s3_bucket_path,
        show_progress_bar=True,
    )


if __name__ == "__main__":
    # Run as "python sentence_transformer.py s3://<bucket_name>/test_mainfn/"
    import sys

    with sbert_stub.run():
        test.call(sys.argv[1])
