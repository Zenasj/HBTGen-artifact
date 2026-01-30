import torch
import torch.cuda.nccl as nccl
import torch.cuda

nGPUs = torch.cuda.device_count()

cpu_inputs = [torch.zeros(128).uniform_().to(dtype=torch.float) for i in range(nGPUs)]
expected = torch.cat(cpu_inputs, 0)

inputs = [cpu_inputs[i].cuda(i) for i in range(nGPUs)]
outputs = [torch.zeros(128 * nGPUs, device=i, dtype=torch.float)
            for i in range(nGPUs)]
nccl.all_gather(inputs, outputs)