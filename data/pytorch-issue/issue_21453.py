import torch
import torchvision

a = torch.rand(1, 3, 299, 299)
m = torchvision.models.inception_v3(transform_input=True).eval()
mt = torch.jit.trace(m, a)