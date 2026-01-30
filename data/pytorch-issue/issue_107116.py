import torch

test_input = torch.randn(2, 3, 4)

print(torch.std(test_input.to('mps'), dim=-1).shape)
print(torch.std(test_input.to('mps'), dim=-2).shape)
print(torch.std(test_input.to('mps'), dim=-3).shape)
print(torch.std(test_input.to('mps'), dim=0).shape)
print(torch.std(test_input.to('mps'), dim=1).shape)
print(torch.std(test_input.to('mps'), dim=2).shape)

print(torch.std(test_input, dim=-1).shape)
print(torch.std(test_input, dim=-2).shape)
print(torch.std(test_input, dim=-3).shape)
print(torch.std(test_input, dim=0).shape)
print(torch.std(test_input, dim=1).shape)
print(torch.std(test_input, dim=2).shape)