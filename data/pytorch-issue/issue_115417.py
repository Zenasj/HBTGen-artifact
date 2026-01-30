import torch.nn as nn

import torch
import platform
import sys

print("Python Version:", sys.version)
print("Torch version: ", torch.__version__)
print("System: ", platform.system())
print("Node: ", platform.node())
print("Release: ", platform.release())
print("Version: ", platform.version())
print("Machine: ", platform.machine())
print("Processor: ", platform.processor())
print("Platform: ", platform.platform())

# some scores tensor
# fmt: off
a = [
    -3.9325406551361084, 0.9592550992965698, 1.1994309425354004, 0.6425036787986755, 1.2429596185684204, 0.5248308777809143, 1.0555568933486938, 0.3594433069229126, 0.7644527554512024, 0.7945529818534851, 0.9265341758728027, 0.2066846489906311, 0.5986100435256958, 0.7194857001304626, 0.6189113855361938, 0.5716232061386108, -0.277407705783844, 0.4133310317993164, -0.06546378135681152, 0.163144052028656, 0.3343402147293091, -0.277407705783844, 0.31046062707901, 0.07031236588954926, 0.1911880075931549, -0.33945488929748535, 0.0906137079000473, -0.01491139829158783, -0.33945488929748535, -0.5645690560340881, 0.1442396193742752, -0.5645690560340881, -0.3931613862514496, -0.8763489723205566, -0.27457353472709656, -0.288480669260025, -0.24146756529808044, -0.07116357982158661, -0.8763489723205566, -0.39812904596328735, -0.268454372882843, -0.8763489723205566, 0.07237496972084045, -0.8763489723205566, -0.3931613862514496, -0.8763489723205566, -0.5904549360275269, -0.6043620705604553, -0.5573489665985107, -0.5843358039855957, -0.5843358039855957, -0.36273038387298584, -0.9802864789962769, -0.5629925727844238, -0.43925708532333374,
]
# fmt: on
a = torch.tensor(a)

# initialize two tensors with negative infinity
x = torch.full((186,), torch.finfo(torch.float32).min)
y = torch.full((256,), torch.finfo(torch.float32).min)

# fill the first 55 elements of x with the scores
x[:55] = a
# fill some middle elements of y with the same scores
y[63 : 63 + 55] = a

# check if the first 55 elements of x and the middle elements of y are equal
x1 = x[:55]
y1 = y[63 : 63 + 55]
print("x1 == y1:", torch.equal(x1, y1))

# compute softmax over both tensors
attn_x = torch.nn.functional.softmax(x, dim=-1)
attn_y = torch.nn.functional.softmax(y, dim=-1)

attn_x1 = attn_x[:55]
attn_y1 = attn_y[63 : 63 + 55]

print("DIFF = ", (attn_x1 - attn_y1).abs().sum())