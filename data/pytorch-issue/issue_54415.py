model = DeformNet().cuda()

with torch.no_grad():
    model_data = model(
                    source_cuda, target_cuda,
                    graph_nodes_cuda, graph_edges_cuda, graph_edges_weights_cuda, graph_clusters_cuda,
                    pixel_anchors_cuda, pixel_weights_cuda,
                    num_nodes_cuda, intrinsics_cuda,
                    evaluate=True, split="test"
    )

# dummy = mask.sum()  # TODO: figure out why there is some kind of lazy-evaluation bug in PyTorch without this line...
                        # del dummy

import torch


def main():

    print("---")
    d = torch.tensor([0.0, 0.5, 0.7, 1.0], device="cuda")
    print(d.mean())

    print("---")

    import open3d as o3d
    import open3d.core as o3c

    voxel_size = 3.0 / 512.0  # voxel resolution in meters
    sdf_trunc = 0.04  # truncation distance in meters
    block_resolution = 16  # 16^3 voxel blocks
    initial_block_count = 100  # initially allocated number of voxel blocks
    device = o3d.core.Device('CUDA:0')

    volume = o3d.t.geometry.TSDFVoxelGrid(
        {
            'tsdf': o3d.core.Dtype.Float32,
            'weight': o3d.core.Dtype.UInt16,
            'color': o3d.core.Dtype.UInt16
        },
        voxel_size=voxel_size,
        sdf_trunc=sdf_trunc,
        block_resolution=block_resolution,
        block_count=initial_block_count,
        device=device
    )

    print("---")

    d = torch.tensor([0.0, 0.5, 0.7, 1.0], device="cuda")
    print(d.mean())

    print("---")

    return 0


if __name__ == "__main__":
    main()