import torch

model = torch.jit.load('checkpoint-10000-embedding.torchscript',
                       map_location='cpu')
model.eval()

x = torch.ones(1, 3, 224, 224)
y = model(x)