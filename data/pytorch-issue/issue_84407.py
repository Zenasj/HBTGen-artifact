import torch

device = torch.device('cuda')
t = torch.tensor([i/50 for i in range(50)])
input = torch.sin(t)
input = torch.stack([input for i in range(64*512)], dim=0)
input = input.to(device)
import time
start = time.time()
stft = torch.stft(input, 16, 10)
print(time.time()-start) # 0.00044
torch.istft(stft, 16, 10) - input
print(time.time()-start) # 0.5130

device = torch.device('cpu')
t = torch.tensor([i/50 for i in range(50)])
input = torch.sin(t)
input = torch.stack([input for i in range(64*512)], dim=0)
input = input.to(device)
import time
start = time.time()
stft = torch.stft(input, 16, 10)
print(time.time()-start) # 0.0399
torch.istft(stft, 16, 10) - input
print(time.time()-start) # 0.1512