py
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = nn.Parameter(torch.rand(2, 2))
        self.layer1 = nn.Linear(4, 4)
        self.layer2 = nn.Linear(4, 4)


def main(rank):
    torch.distributed.init_process_group(
        backend="nccl", init_method="tcp://127.0.0.1:50001", rank=rank, world_size=2
    )
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    
    # Regular PyTorch state dict on CPU
    model = CustomModel()
    state_dict = model.state_dict()

    with torch.device("meta"):
        model = CustomModel()

    device_mesh = init_device_mesh("cuda", mesh_shape=(2,), mesh_dim_names=("tp",))
    plan = {"layer1": ColwiseParallel(), "layer2": ColwiseParallel()}
    parallelize_module(model, device_mesh, plan)
    model.to_empty(device=device)
    
    # Load state dict into DTensor model
    state_dict_options = StateDictOptions(
        broadcast_from_rank0=True,
        full_state_dict=True,
        strict=True,  # gets ignored at the moment
    )
    
    # Strict loading doesn't work! This should error!
    del state_dict["parameter"]
    set_model_state_dict(model, state_dict, options=state_dict_options)



if __name__ == "__main__":
    mp.spawn(main, nprocs=2)