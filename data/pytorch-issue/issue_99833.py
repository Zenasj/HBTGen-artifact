import torch

def recursive_identity(x):
    length = x.shape[-1]
    if length <= 1: return x

    x_0 = x[..., :length//2]
    x_1 = x[..., length//2:]

    y_1 = recursive_identity(x_1)
    y_0 = x_0

    y = torch.cat([y_0, y_1], dim=-1)
    return y

def test(L):
    a = 1 + torch.arange(L)

    csum = recursive_identity(a)
    print(csum)

    # Compile function
    recursive_identity_opt = torch.compile(recursive_identity)
    csum = recursive_identity_opt(a)
    print(csum)


if __name__ == "__main__":
    test(4)