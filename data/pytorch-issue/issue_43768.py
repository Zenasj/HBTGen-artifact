import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

x = torch.randn(0, 128)
# Works as expected as there is nothing 0x64
torch.multinomial(x, 64, replacement = True)
# Crashes
torch.multinomial(x.to(device), 64, replacement = True)