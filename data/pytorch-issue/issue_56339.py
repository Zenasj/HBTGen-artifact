import torch

example_input = torch.rand(1, 3, 256, 192) 
traced_model = torch.jit.trace(model, example_input)