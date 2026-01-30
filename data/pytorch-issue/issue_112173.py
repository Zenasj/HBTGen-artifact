import torch.nn as nn

import nvidia_smi
import torch
from tqdm.auto import tqdm


def memory():
    nvidia_smi.nvmlInit()
    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    freem = []
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        freem.append(100 * info.free / info.total)
    nvidia_smi.nvmlShutdown()
    return freem


class MyToyPytorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1408)
        self.to("cuda")

    def forward(self, image_features):
        reward = self.linear(image_features)
        return reward


def f():
    if getattr(f, "model", None) is None:
        f.model = MyToyPytorchModel()
    model = f.model
    inputs = torch.randn(2, 3).to("cuda")
    image_features = model(inputs)
    image_features /= image_features.norm(dim=-1, keepdim=True)  # memory leak
    # image_features = image_features / image_features.norm(dim=-1, keepdim=True) # no memory leak
    reward = (image_features[0] * image_features[1:]).sum(-1)
    return reward.detach().cpu().numpy()


with tqdm(range(10**6)) as pbar:
    for _ in pbar:
        pbar.set_postfix({"reward": f(), "memory": memory()})

3
import torch

model = torch.nn.Linear(4, 1024).cuda()

for i in range(10 ** 6):
    inputs = torch.randn(2, 4).to("cuda")
    image_features = model(inputs)
    image_features /= image_features.norm(dim=-1, keepdim=True)

3
import torch

model = torch.nn.Linear(4, 1024).cuda()
leak = True

for i in range(10 ** 6):
    inputs = torch.randn(2, 4).to("cuda")
    x = model(inputs)
    if leak:
        print("\rpow", end='')
        x += torch.pow(x, 2)
    else:
        print("\rsqrt", end='')
        x += torch.sqrt(x)