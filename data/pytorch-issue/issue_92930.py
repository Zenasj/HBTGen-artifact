import torch

with FakeTensorMode():
    device_mesh = DeviceMesh("cuda", torch.arange(4))

with FakeTensorMode():
    device_mesh = DeviceMesh("cuda", torch.arange(4))