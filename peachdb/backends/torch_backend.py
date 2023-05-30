import numpy as np
import torch

from peachdb.backends.backend_base import BackendBase


def _check_dims(query_embed: torch.Tensor, embeds: torch.Tensor):
    if query_embed.dim() == 1:
        query_embed = query_embed.unsqueeze(0)
    elif query_embed.dim() == 2:
        if query_embed.size(0) != 1:
            raise ValueError("query_embed should be a vector or a matrix with one row")
    else:
        raise ValueError("query_embed should be a vector or a matrix with one row")

    if embeds.dim() != 2:
        raise ValueError("embeds should be a 2-D matrix")

    return query_embed, embeds


def l2(query_embed: torch.Tensor, embeds: torch.Tensor) -> torch.Tensor:
    """
    Calculate l2 distance between a query embedding and a set of embeddings.
    """
    query_embed, embeds = _check_dims(query_embed, embeds)

    return torch.norm(query_embed - embeds, dim=1)


def cosine(query_embed: torch.Tensor, embeds: torch.Tensor) -> torch.Tensor:
    """
    Can be used to compute cosine "distance" between any number of query embeddings
    and a set of embeddings.
    result[i, j] = 1 - torch.dot(query_embed[i], embeds[j])
    """
    query_embed, embeds = _check_dims(query_embed, embeds)

    return (
        1
        - torch.mm(query_embed, embeds.t())
        / (torch.norm(query_embed, dim=1).unsqueeze(1) * torch.norm(embeds, dim=1).unsqueeze(0))
    )[0]


class TorchBackend(BackendBase):
    def __init__(
        self,
        embeddings_dir: str,
        metadata_path: str,
        embedding_generator: str,
        distance_metric: str,
        id_column_name: str,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to use TorchDB")

        super().__init__(
            embeddings_dir=embeddings_dir,
            metadata_path=metadata_path,
            embedding_generator=embedding_generator,
            distance_metric=distance_metric,
            id_column_name=id_column_name,
        )
        self.device = torch.device("cuda")
        self._embeddings = torch.from_numpy(self._embeddings).to(self.device)  # Ensure the tensor is on the GPU

    def _process_query(self, query_embedding, top_k: int = 5):
        """Compute query embedding, calculate distance of query embedding and get top k."""
        query_embedding = torch.from_numpy(query_embedding).to(self.device)

        print("Calculating distances...")
        distances = (
            l2(query_embedding, self._embeddings)
            if self._distance_metric == "l2"
            else cosine(query_embedding, self._embeddings)
        )

        print("Getting top results...")
        results = torch.argsort(distances)[:top_k].cpu().numpy()
        return self._ids[results], distances[results].cpu().numpy()


if __name__ == "__main__":
    import scipy.spatial.distance as scipy_distance
    from sentence_transformers.util import cos_sim as st_cos_sim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dim in [3, 384, 1536]:
        a = torch.rand(dim, device=device)
        b = torch.rand(3, dim, device=device)

        # cosine
        cosine_result = cosine(a, b)
        for i in range(b.shape[0]):
            np.testing.assert_allclose(
                scipy_distance.cosine(a.cpu().numpy(), b[i].cpu().numpy()),
                cosine_result[i].cpu().numpy(),
                rtol=1e-4,
            )
            np.testing.assert_allclose(
                1 - st_cos_sim(a.cpu().numpy(), b[i].cpu().numpy()).numpy(),
                cosine_result[i].cpu().numpy(),
                rtol=1e-4,
            )

        # l2
        l2_result = l2(a, b)
        for i in range(b.shape[0]):
            np.testing.assert_allclose(
                scipy_distance.euclidean(a.cpu().numpy(), b[i].cpu().numpy()),
                l2_result[i].cpu().numpy(),
                rtol=1e-4,
            )
