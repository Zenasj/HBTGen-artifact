import torch
dummy = torch.randn(1, 3, 640, 640)
model = torch.jit.load("yolos.int8.torchscript.pt")

model(dummy)  # this will succeed

model(dummy) # this will fail