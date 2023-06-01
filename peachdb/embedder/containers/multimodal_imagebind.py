import modal

from peachdb.embedder.containers.base import EmbeddingModelBase, base_container_image, modal_compute_spec_decorator
from peachdb.embedder.models.multimodal_imagebind import ImageBindModel

IMAGEBIND_BATCH_SIZE = 1024

imagebind_stub = modal.Stub("ImageBind")


def download_model():
    ImageBindModel.download_model()


image = base_container_image.run_function(download_model)


@modal_compute_spec_decorator(stub=imagebind_stub, image=image)
class ImageBindEmbdedder(EmbeddingModelBase):
    def __enter__(self):
        self.model = ImageBindModel()

    def _calculate_embeddings(self, texts, show_progress_bar: bool):
        return self.model.encode(texts, IMAGEBIND_BATCH_SIZE, show_progress_bar)

    @modal.method()
    def calculate_embeddings(self, texts, ids, s3_bucket: str, show_progress_bar: bool):
        return super().calculate_embeddings(texts, ids, s3_bucket, show_progress_bar)


# We have a function here instead of putting it into __main__ so that `modal shell` works
@imagebind_stub.function(image=image)
def test(s3_bucket_path: str):
    ib = ImageBindEmbdedder()
    embeddings = ib.calculate_embeddings.call(
        ["hello", "world"],
        [1, 2],
        s3_bucket_path,
        show_progress_bar=True,
    )


if __name__ == "__main__":
    # Run as "python multimodal_imagebind.py s3://<bucket_name>/test_mainfn/"
    import sys

    with imagebind_stub.run():
        test.call(sys.argv[1])
