import torch

world_size = 1
device_type="cuda"

all_gather_output = torch.randn((288), device=device_type)
all_gather_output = all_gather_output.view(world_size, -1)

out = [torch.randn((1, 256), device=device_type), torch.randn((1, 32), device=device_type)]
all_gather_input_split_sizes = [256, 32]

print(f"before version: {out[0]._version}")
torch.split_with_sizes_copy(all_gather_output, all_gather_input_split_sizes, dim=1, out=out)
print(f"after version: {out[0]._version}")