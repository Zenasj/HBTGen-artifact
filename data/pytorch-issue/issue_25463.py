import torch
import torch.distributed as dist

dist.init_process_group("gloo")

tensor = torch.tensor([dist.get_rank()], dtype=torch.int32)
if dist.get_rank() == 0:
    output = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.gather(tensor=tensor, gather_list=output, dst=0)
    print(output)
else:
    dist.gather(tensor=tensor, gather_list=[], dst=0)