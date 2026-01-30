import torch
window_length = 10
window1 = torch.bartlett_window(window_length, requires_grad=True)
window2 = window1.type(torch.long)
window_np = window2.numpy()
print(window1)
print(window2)
print(window_np)

import torch
window_length = 10
window1 = torch.bartlett_window(window_length, requires_grad=True)
window_np = window1.numpy()
print(window1)
print(window_np)

import torch
window_length = 10
window1 = torch.bartlett_window(window_length, requires_grad=True)
window2 = window1.to(dtype=torch.long)
window_np = window2.numpy()
print(window1)
print(window2)
print(window_np)