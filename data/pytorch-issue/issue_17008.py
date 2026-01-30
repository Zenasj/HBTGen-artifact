import random

x[i], x[j] = x[j], x[i]

import torch
x = torch.randn(4)  # tensor([-0.7905, -0.3574, -0.0493, -0.3152])
x[0], x[1] = x[1], x[0]
print(x)  # tensor([-0.3574, -0.3574, -0.0493, -0.3152])

x = torch.randn(4, 4)
assert not (x[0] == x[1]).all()
x[0], x[1] = x[1], x[0]
assert not (x[0] == x[1]).all()

import numpy as np
x = np.random.randn(4)  # array([ 0.11462874, -1.3748384 ,  0.24256925,  1.01256207])
x[0], x[1] = x[1], x[0]
print(x)  # array([-1.3748384 ,  0.11462874,  0.24256925,  1.01256207])

import numpy as np
x = np.random.randn(4)
print(x)
# array([ 1.78025159,  0.24798579,  0.29656804, -1.06337399])
y = x[0]
y += 1
print(x)
# array([ 1.78025159,  0.24798579,  0.29656804, -1.06337399])
print(y)
# 2.7802515871964264