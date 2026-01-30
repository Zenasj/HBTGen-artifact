import torch

torch.manual_seed(0)

for N in range(270, 350):
    W = torch.qr(torch.FloatTensor(N, N).normal_())[0]
    print('C = {}, det on cpu/gpu = {:.6f} / {:.6f}'.format(N,W.det(),W.cuda().det()))