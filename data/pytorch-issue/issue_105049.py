# torch.randint(1, 10, (2,), dtype=torch.int32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, input_tensor):
        world_size = int(input_tensor[0].item())
        nGPUs = int(input_tensor[1].item())
        orig_rank_to_GPU = self.original_helper(world_size, nGPUs)
        corr_rank_to_GPU = self.corrected_helper(world_size, nGPUs)
        return self.compare_dicts(orig_rank_to_GPU, corr_rank_to_GPU)
    
    def original_helper(self, world_size, nGPUs):
        visible_devices = list(range(nGPUs))
        nGPUs_per_process = 1
        if world_size > nGPUs:
            nGPUs_per_process = nGPUs // world_size
        rank_to_GPU = {}
        for i in range(world_size):
            start = i * nGPUs_per_process
            end = (i + 1) * nGPUs_per_process
            devices = visible_devices[start:end]
            rank_to_GPU[i] = devices
        return rank_to_GPU
    
    def corrected_helper(self, world_size, nGPUs):
        visible_devices = list(range(nGPUs))
        nGPUs_per_process = 1
        if world_size < nGPUs:
            nGPUs_per_process = nGPUs // world_size
        elif world_size > nGPUs:
            nGPUs_per_process = 0  # As per team's comment for world_size > nGPUs
        rank_to_GPU = {}
        for i in range(world_size):
            start = i * nGPUs_per_process
            end = (i + 1) * nGPUs_per_process
            devices = visible_devices[start:end]
            rank_to_GPU[i] = devices
        return rank_to_GPU
    
    def compare_dicts(self, orig, corr):
        if len(orig) != len(corr):
            return torch.tensor(0, dtype=torch.int32)
        for key in orig:
            if key not in corr or orig[key] != corr[key]:
                return torch.tensor(0, dtype=torch.int32)
        return torch.tensor(1, dtype=torch.int32)

def my_model_function():
    return MyModel()

def GetInput():
    # Random integers between 1-9 for world_size and nGPUs
    world_size = torch.randint(1, 10, (1,)).item()
    nGPUs = torch.randint(1, 10, (1,)).item()
    return torch.tensor([world_size, nGPUs], dtype=torch.int32)

