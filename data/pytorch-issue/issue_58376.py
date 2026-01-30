import torch

memory_init = torch.cuda.memory_stats_as_nested_dict(0)["allocated_bytes"]["all"]["current"]
try:
    t = torch.zeros(size=(1000 ,1000), device=torch.device('cuda:0'))
    memory_used = (torch.cuda.memory_stats_as_nested_dict(0)["allocated_bytes"]["all"]["current"] - memory_init)
    del t
except:
    memory_used = None # The tensor/object does not fit on the currently selected device.
torch.cuda.empty_cache()