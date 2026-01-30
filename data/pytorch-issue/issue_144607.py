import torch

@torch.compile
def foobar(x):
    return x * 2

def test(device):
    foobar(torch.empty((1, 16, 128, 128), device = device))
    foobar(torch.empty((1, 32, 64, 64), device = device))

# OK
test("cuda")
print("cuda ok")

# Fails
test("meta")
print("meta ok")