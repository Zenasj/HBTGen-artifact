import torch, sys
print('Torch version:', torch.__version__)
print('sys.version', sys.version)
for n_rows in [
        0b01000000000000000000001,
        0b01000000000000000000010,
        0b10000010010000010001110,
        0b11000000000000000000000
        ]:
    a=torch.ones(n_rows,2).float().cuda()
    b=torch.ones(2,2).float().cuda()
    print((torch.mm(a, b) - 2).abs().max().item())