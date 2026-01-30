import torch
from torch._subclasses.fake_tensor import FakeTensorMode
with FakeTensorMode():
    with torch.inference_mode():
        torch.tensor(32.).item()