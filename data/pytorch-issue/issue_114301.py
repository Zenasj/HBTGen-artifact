# torch.rand(1, dtype=torch.float32)
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _coalescing_manager

class MyModel(torch.nn.Module):
    def forward(self, tensor):
        pg = dist.group.WORLD
        device = torch.cuda.current_device()
        tensor = tensor.to(device)

        coalescing_success = False
        coalescing_result = None
        try:
            with _coalescing_manager(pg, device, async_ops=True) as cm:
                op = dist.all_reduce(tensor, op=dist.ReduceOp.AVG, group=pg, async_op=True)
                cm.wait()
            coalescing_success = True
            coalescing_result = tensor.clone()
        except:
            pass

        normal_success = False
        normal_result = None
        try:
            dist.all_reduce(tensor, op=dist.ReduceOp.AVG, group=pg)
            normal_success = True
            normal_result = tensor.clone()
        except:
            pass

        # Return True if the two approaches differ in outcome
        if coalescing_success != normal_success:
            return torch.tensor([True], dtype=torch.bool)
        elif coalescing_success and normal_success:
            return torch.tensor([not torch.allclose(coalescing_result, normal_result)], dtype=torch.bool)
        else:
            return torch.tensor([False], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, dtype=torch.float32)

