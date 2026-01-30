import torch

real_inputs =  torch.tensor([[11, 22], [33, 44]], dtype=torch.float32)
torch.package.PackageExporter(real_inputs, './test.zip')