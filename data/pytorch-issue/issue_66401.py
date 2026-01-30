import torch
r = [1.0, 8.0, 23.0, 52.0]
o = [0.0, 0.0, 0.0, 0.0]
B = torch.tensor([r, o, r, o])
# gives
# tensor([[ 1.,  8., 23., 52.],
#         [ 0.,  0.,  0.,  0.],
#         [ 1.,  8., 23., 52.],
#         [ 0.,  0.,  0.,  0.]])
print(B.stride()) # (4, 1)
print(B[::2])
# tensor([[ 1.,  8., 23., 52.],
#         [ 1.,  8., 23., 52.]])
print(B[::2].stride()) # (8, 1)
# so that's why we pick ldb = 8