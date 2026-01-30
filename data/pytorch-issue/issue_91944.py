import torch

def foo(x):
    return torch.randn(x.size())

opt_foo = torch.compile(foo)
x = opt_foo(torch.randn(10000))
print(x)
print("mean: ", torch.mean(x)) # 1.7567, but E[X] should be close to 0
print("std: ", torch.std(x)) # 2.3843e-07 (i.e., is not random), but SD[X] should be close to 1