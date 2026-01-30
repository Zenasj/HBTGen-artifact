import torch

model = torch.jit.load("sample.pt")
model.eval()
model.cuda()

sample_tensor = torch.randn(1, 3, 244, 244)

model(sample_tensor.cuda())