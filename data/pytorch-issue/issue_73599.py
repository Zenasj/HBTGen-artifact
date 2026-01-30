import torch
from torch.utils import mobile_optimizer
model = torch.jit.load("Mclaren_traced.pt")
vk_model = mobile_optimizer.optimize_for_mobile(model, backend="vulkan")
print(vk_model.graph)