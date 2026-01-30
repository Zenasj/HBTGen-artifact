import torch.nn as nn

from typing import Optional

import torch
from torch import nn


class CreateVoxelGrid(nn.Module):
    def __init__(self, shape: tuple[int, int, int, int]) -> None:
        super().__init__()
        self.grid_shape = shape

    def forward(
        self,
        voxel_features: torch.Tensor,
        indices: torch.Tensor,
        voxel_features_mask: Optional[torch.Tensor] = None,
        indices_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        grid = voxel_features.new_zeros(self.grid_shape)

        if voxel_features_mask is not None:
            voxel_features = voxel_features[voxel_features_mask]
        if indices_mask is not None:
            indices = indices[indices_mask]
        grid[indices[:, 0], indices[:, 1], indices[:, 2]] = voxel_features
        return grid


class FixedShapeUnique(nn.Module):
    def forward(
        self,
        tensor: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            mask = torch.ones(tensor.shape[0], dtype=torch.bool, device=tensor.device)

        output = torch.zeros_like(tensor)
        valid = torch.zeros_like(mask)

        unique_tensor = torch.unique(tensor[mask], dim=0)

        output[: unique_tensor.shape[0]] = unique_tensor
        valid[: unique_tensor.shape[0]] = True

        return output, valid


class Network(nn.Module):
    def __init__(self, grid_shape: tuple[int, int, int, int]) -> None:
        super().__init__()

        self.unique = FixedShapeUnique()
        self.voxel_grid = CreateVoxelGrid(grid_shape)

    def forward(
        self,
        voxel_features: torch.Tensor,
        indices: torch.Tensor,
        voxel_features_mask: Optional[torch.Tensor] = None,
        indices_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        indices, indices_mask = self.unique(indices, mask=indices_mask)  # <- the million dollar question
        return self.voxel_grid(
            voxel_features, indices, voxel_features_mask=voxel_features_mask, indices_mask=indices_mask
        )


def main():
    torch.manual_seed(24)

    channels = 8
    n_occupied_voxels = 20
    voxel_features = torch.randn(n_occupied_voxels, channels)

    batch_size = 1
    grid_shape = (batch_size, 256, 256, channels)
    indices = torch.stack([torch.randint(size, size=(n_occupied_voxels,)) for size in grid_shape], dim=1)

    voxel_features_mask = torch.rand(voxel_features.shape[0]) > 0.5
    # just creating a new mask with the same number of True elements
    indices_mask = torch.flipud(voxel_features_mask)

    model = Network(grid_shape)
    model(voxel_features, indices, voxel_features_mask=voxel_features_mask, indices_mask=indices_mask)

    path = "/tmp/playground.onnx"

    torch.onnx.export(
        model=model.eval(),
        args=(voxel_features, indices, {"voxel_features_mask": voxel_features_mask, "indices_mask": indices_mask}),
        f=path,
        opset_version=15,
        input_names=["voxel_features", "indices", "voxel_features_mask", "indices_mask"],
        export_modules_as_functions={FixedShapeUnique, CreateVoxelGrid},
    )


if __name__ == "__main__":
    main()

class FixedShapeUnique(nn.Module):
    def forward(
        self,
        tensor: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            mask = torch.ones(tensor.shape[0], dtype=torch.bool, device=tensor.device)

        output = torch.zeros_like(tensor)
        valid = torch.zeros_like(mask)

        unique_tensor = torch.unique(tensor[mask], dim=0)

        output[: unique_tensor.shape[0]] = unique_tensor
        valid[: unique_tensor.shape[0]] = True

        return output, valid