import torch

tp_size = 2
dp_mesh_dim = 0
mesh = DeviceMesh("cuda", torch.arange(dist.get_world_size()).view(tp_size, -1))
dp_pg = mesh.get_dim_groups()[dp_mesh_dim]
if dist.get_rank() == 0:
    print(f"mesh: {mesh}")
    print(
        f"dp_pg size: {dp_pg.size()} "
       f"ranks: {torch.distributed.distributed_c10d.get_process_group_ranks(dp_pg)}"
    )