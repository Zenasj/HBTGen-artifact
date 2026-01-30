import torch, gzip
a = torch.load(gzip.open('S.pt7.gz'))

print(a.device)
# cuda:0

a.eig()
#Intel MKL ERROR: Parameter 2 was incorrect on entry to SGEHD2.

#(tensor([[ 1.0000,  0.0000],
#        [ 0.5825,  0.0000],
#        [ 0.1131,  0.0000],
#        ...,
#        [ 0.0000,  0.0000],
#        [ 0.0000, -0.0000],
#        [ 0.0000,  0.0000]], device='cuda:0'), tensor([], device='cuda:0'))