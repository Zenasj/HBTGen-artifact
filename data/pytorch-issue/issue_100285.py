import random

import torch
import numpy as np
image_tensor = torch.tensor(np.random.rand(256, 256, 3) * 255, dtype=torch.float32, device="mps")
image_tensor
'''
-[_MTLCommandBuffer addCompletedHandler:]:867: failed assertion `Completed handler provided after commit call'
'''

import torch
import threading

def f1(x):
    z=torch.nonzero(x)

x=torch.rand(3, 3, device="mps")

t1 = threading.Thread(target=f1, args=(x,))
t2 = threading.Thread(target=f1, args=(x,))
t1.start()
t2.start()