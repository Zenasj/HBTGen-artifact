import torch
import itertools

print(torch.__version__)

def test(dtype, device):
    print(dtype, device)

    if dtype == torch.float:
        num = 3e38  # 3.4e38
    elif dtype == torch.half:
        num = 40000 # 65535

    a = torch.tensor([num, num, num], dtype=dtype, device=device)
    print('data', a)

    b = a.mean()
    print('mean', b)

    if dtype == torch.half and device == 'cpu':
        # RuntimeError: _th_var not supported on CPUType for Half
        print()
        return

    c = a.var()
    print('var', c)

    print()


for dtype, device in itertools.product(
    [torch.half, torch.float],
    ['cpu', 'cuda']):

    test(dtype, device)