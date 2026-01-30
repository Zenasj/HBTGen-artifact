import torch
x_cpu = torch.tensor([3388.]).half().to("cpu")
x_gpu = torch.tensor([3388.]).half().to("cuda:0")
scale = torch.tensor([524288.0])
print(x_cpu.div(scale), x_gpu.div(scale.to("cuda:0")))