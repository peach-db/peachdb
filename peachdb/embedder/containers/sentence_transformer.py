import modal

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

    def _calculate_embeddings(self, texts, show_progress_bar: bool):
        return self.model.encode(
            texts,
            batch_size=SENTENCE_TRANSFORMER_BATCH_SIZE,
            show_progress_bar=show_progress_bar,
        )

    @modal.method()
    def calculate_embeddings(self, texts, ids, s3_bucket: str, show_progress_bar: bool):
        return super().calculate_embeddings(texts, ids, s3_bucket, show_progress_bar)


# We have a function here instead of putting it into __main__ so that `modal shell` works
@sbert_stub.function(image=image)
def test(s3_bucket_path: str):
    st = SentenceTransformerEmbedder()
    embeddings = st.calculate_embeddings.call(
        ["hello", "world"],
        [1, 2],
        s3_bucket_path,
        show_progress_bar=True,
    )


if __name__ == "__main__":
    # Run as "python sentence_transformer.py s3://<bucket_name>/test_mainfn/"
    import sys

    with sbert_stub.run():
        test.call(sys.argv[1])
