import torch
import torch.nn as nn


block = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.Conv2d(8, 1, 3, padding=1))
with torch.no_grad():
    gx = block(torch.randn(1,1,5,5))

print(gx)

tensor([[[[        nan,         nan, -5.0249e+27,         nan,         nan],
          [        nan,         nan, -1.0987e+28,         nan,         nan],
          [        nan,         nan,  2.0282e+36, -2.5592e+36,  8.3669e+35],
          [        nan,         nan, -4.8736e+36, -7.4338e+35,  4.1086e+36],
          [        nan,         nan, -2.9828e+36, -2.1171e+36,  3.1350e+36]]]])

block = nn.Conv2d(1, 1, 3, padding=1)
with torch.no_grad():
    gx = block(torch.randn(1,1,5,5))

print(gx)

'''output
tensor([[[[ 0.5557,  0.4501, -0.1535, -0.8888, -0.8810],
          [ 0.6834,  0.4136, -0.0093, -0.7025, -1.4664],
          [ 0.7698,  0.6245, -0.3802, -1.1841, -1.5778],
          [ 0.6865,  0.5524, -1.0033, -1.0162, -1.3443],
          [ 0.4791,  0.3035, -0.3653, -0.7445, -1.0624]]]])
'''