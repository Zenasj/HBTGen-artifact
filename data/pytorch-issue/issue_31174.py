import torch

print('Version', torch.__version__)
# 1.3.0 for me

f = torch.arange(4*4, dtype=torch.float).view(4, 4)
print('f\n', f)
# f
#  tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11.],
#         [12., 13., 14., 15.]])
g = torch.zeros((4, 4), dtype=torch.bool)
g[:, :1] = True
print('g\n', g)
# g
#  tensor([[ True, False, False, False],
#         [ True, False, False, False],
#         [ True, False, False, False],
#         [ True, False, False, False]])
print('Wrong: f*g.t()\n', f * (g.t()))
# Wrong: f*g.t()
#  tensor([[ 0.,  0.,  0.,  0.],
#         [ 4.,  0.,  0.,  0.],
#         [ 8.,  0.,  0.,  0.],
#         [12.,  0.,  0.,  0.]])
print('Correct: f*g.t().clone()\n', f * (g.t().clone()))
# Correct: f*g.t().float()
#  tensor([[0., 1., 2., 3.],
#         [0., 0., 0., 0.],
#         [0., 0., 0., 0.],
#         [0., 0., 0., 0.]])
print('Correct: f*g.t().float()\n', f * (g.t().float()))
# Correct: f*g.t().float()
#  tensor([[0., 1., 2., 3.],
#         [0., 0., 0., 0.],
#         [0., 0., 0., 0.],
#         [0., 0., 0., 0.]])
print('Correct: f*g.float().t() \n', f * (g.float().t()))
# Correct: f*g.float().t()
#  tensor([[0., 1., 2., 3.],
#         [0., 0., 0., 0.],
#         [0., 0., 0., 0.],
#         [0., 0., 0., 0.]])