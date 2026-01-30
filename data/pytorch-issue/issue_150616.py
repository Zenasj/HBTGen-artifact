import torch
window_length = 10
window1 = torch.bartlett_window(window_length, requires_grad=True)
window2 = window1.type(torch.long)
print(window1)
print(window2)

import torch
window_length = 10
window1 = torch.bartlett_window(window_length, requires_grad=True)
window2 = window1.type(torch.LongTensor)
print(window1)
print(window2)

import torch
window_length = 10
window1 = torch.bartlett_window(window_length, dtype=torch.float, requires_grad=True)
print(window1)

import torch
window_length = 10
window1 = torch.bartlett_window(window_length, dtype=torch.long, requires_grad=True)
print(window1)