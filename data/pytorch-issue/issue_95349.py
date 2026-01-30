import torch

def test(input):
    return torch.Tensor.aminmax(input,dim=0)

# device = "cpu"
device = "cuda"
input = torch.ones([10, 0], dtype=torch.float32)
x=test(input.to(device))
print(x)