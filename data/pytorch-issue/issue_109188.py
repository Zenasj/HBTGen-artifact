import torch.nn as nn

3
import torch
import copy

print(torch.__version__)
torch.manual_seed(0)

layernorm = torch.nn.LayerNorm(12, eps=0.0, elementwise_affine = False).bfloat16()
layernorm_cuda = copy.deepcopy(layernorm).cuda()

input = torch.randn((2, 12), dtype=torch.bfloat16)
input_cuda = input.cuda()

output_cpu = layernorm(input)
output_cuda = layernorm_cuda(input_cuda)
print("CPU", output_cpu)
print("CUDA", output_cuda)
print("max diff", (output_cpu-output_cuda.cpu()).abs().max())