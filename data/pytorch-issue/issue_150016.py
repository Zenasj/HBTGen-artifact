import torch
import torch.nn as nn
torch.set_default_device('cuda:0')

def test(layer, x):
    try:
        layer(x)
        return True, None
    except RuntimeError as e:
        return False, e

layer = nn.Linear(2048, 2048)
x = torch.randn((1, 32768, 2048))[:, 0]
print(test(layer, x))

# test1 ok
print(1, test(nn.Linear(2048, 2048), torch.randn((1, 32767, 2048))[:, 0]))

# test2 ok
print(2, test(nn.Linear(2047, 2047), torch.randn((1, 32784, 2047))[:, 0]))

# test3 fail
print(3, test(nn.Linear(2047, 2047), torch.randn((1, 32785, 2047))[:, 0]))

# test4 ok
print(4, test(nn.Linear(1024, 1024), torch.randn((1, 65535, 1024))[:, 0]))

# test5 fail
print(5, test(nn.Linear(1024, 1024), torch.randn((1, 65536, 1024))[:, 0]))

# test6 ok
print(6, test(nn.Linear(2048, 2048), torch.randn((16, 32767, 2048))[:, 0]))

# test7 fail
print(7, test(nn.Linear(2048, 2048), torch.randn((16, 32768, 2048))[:, 0]))

# test8 fail
layer = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU())
x = torch.randn((1, 32768, 2048))[:, 0]
print(8, test(layer, x))

# test9 ok
layer = nn.Sequential(nn.ReLU(), nn.Linear(2048, 2048))
x = torch.randn((1, 32768, 2048))[:, 0]
print(9, test(layer, x))