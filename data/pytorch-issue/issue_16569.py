import numpy as np
import random

np.random.seed(2)
a = np.random.rand(1, 10000).astype('float32')
d = a.mean() - np.ascontiguousarray(a.reshape(50, 5, 40).transpose((1, 0, 2))).mean()

print(d)

# gives -5.9604645e-08