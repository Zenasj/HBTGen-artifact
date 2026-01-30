import torch

device = 'cuda'
model = BiRNN(10, 10, 10, 10, 10, 10, 0.5).to(device)
x = torch.randint(0, 10, (10, 10)).to(device)
output = model(x)