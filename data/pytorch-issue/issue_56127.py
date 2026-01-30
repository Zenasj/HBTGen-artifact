import torch
import torch.profiler as profiler

with profiler.profile() as prof:
	torch.add(1, 1)

print(prof.key_averages().table())
import pdb ; pdb.set_trace()