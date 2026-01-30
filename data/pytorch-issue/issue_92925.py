import torch.nn as nn
import torch.nn.functional as F

import torch
from einops import rearrange, repeat
from torch import einsum, nn
from torch.nn import functional as F


def sum_squares(t, dim=-1):
    return (t**2).sum(dim=dim)


class VectorQuantization(nn.Module):
    def __init__(self, *, num_clusters, num_heads, dim_per_head, decay=0.999, epsilon=1e-6):
        super().__init__()
        self.decay = decay
        self.epsilon = epsilon
        self.num_heads = num_heads
        self.dim_per_head = dim_per_head
        self.num_clusters = num_clusters

        self.register_buffer("means", torch.randn(num_heads, num_clusters, dim_per_head))

    def forward(self, x, mask=None):
        h, dim_head, num_clusters, eps, decay, means = (
            self.num_heads,
            self.dim_per_head,
            self.num_clusters,
            self.epsilon,
            self.decay,
            self.means,
        )
        assert x.shape[-1] == (
            h * dim_head
        ), f"input embedding feature dimension must be {h * dim_head}"

        # split heads from input
        x = rearrange(x, "b n (h d) -> b n h d", h=h)

        # get distance of input embeddings from means
        dists = (
            rearrange(sum_squares(x), "b n h -> b n h 1")
            - 2 * einsum("b n h d, h k d -> b n h k", x, means)
            + rearrange(sum_squares(means), "h k -> 1 1 h k")
        )

        # get cluster ids
        cluster_ids = dists.argmin(dim=-1)

        if self.training:
            # get one hot, for calculating number of matches per mean
            nearest_one_hot = F.one_hot(cluster_ids, num_classes=num_clusters)
            per_cluster_count = nearest_one_hot.sum(dim=(0, 1))

            # sum of the input per each closest centroid.
            sum_x = einsum("b n h k, b n h d -> h k d", nearest_one_hot.to(x.dtype), x)

            # calculate new means
            new_means = sum_x / (eps + rearrange(per_cluster_count, "... -> ... 1"))

            # exponential moving average
            updated_means = (1.0 - decay) * new_means + decay * means
            self.means.data.copy_(updated_means)

        return cluster_ids


input = torch.randn(4, 32, 512)
model = VectorQuantization(
    num_clusters=512,
    num_heads=32,
    dim_per_head=16,
)

model(input)

model = torch.compile(model)

model(input)