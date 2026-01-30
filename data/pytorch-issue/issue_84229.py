import torch
model = torch.hub.load('ultralytics/yolov3', 'yolov3')
device = torch.device('mps')
model.to(device)
img = 'https://ultralytics.com/images/zidane.jpg'
model(img)

coords[:,[0,2]] -= pad[0] 
coords[:,[1,3]] -= pad[1]

tensor([[ 71.47987, 112.43367, 531.28778, 369.58679],
        [371.92389,  33.11592, 563.16974, 368.19769],
        [216.71069, 228.54417, 260.64929, 370.04932],
        [495.18076, 170.43976, 511.94131, 220.99673]], device='mps:0')

tensor([[ 71.47987, 112.43367, 531.28778, 369.58679],
        [563.16974, 368.19769, 216.71069, 228.54417],
        [495.18076, 170.43976, 511.94131, 220.99673],
        [495.18076, 170.43976, 511.94131, 220.99673]], device='mps:0')

tensor([[ 71.47987, 100.43367, 531.28778, 357.58679],
        [216.71069, 216.54417, 495.18076, 158.43976],
        [495.18076, 158.43976, 511.94131, 208.99673],
        [495.18076, 170.43976, 511.94131, 220.99673]], device='mps:0')

tensor([[ 71.47993, 100.43364, 531.28772, 357.58685],
        [371.92389,  21.11584, 563.16980, 356.19778],
        [216.71071, 216.54416, 260.64926, 358.04935],
        [495.18079, 158.43976, 511.94128, 208.99673]])

import numpy as np
from torch import tensor
import torch

import sys

# you'll need to change this to whatever is appropriate on your system
sys.path.append('/Users/gad/.cache/torch/hub/ultralytics_yolov3_master')
from utils.general import scale_coords

print('numpy', np.__version__)
print('pytorch', torch.__version__)

device = torch.device("mps")

coords = np.array([[7.14799e+01, 1.12434e+02, 5.31288e+02, 3.69587e+02, 9.36926e-01, 0.00000e+00],
        [3.71924e+02, 3.31158e+01, 5.63170e+02, 3.68198e+02, 9.16234e-01, 0.00000e+00],
        [2.16711e+02, 2.28544e+02, 2.60649e+02, 3.70049e+02, 8.61753e-01, 2.70000e+01],
        [4.95181e+02, 1.70440e+02, 5.11941e+02, 2.20997e+02, 3.32481e-01, 2.70000e+01]],dtype=np.float32)

img1_shape = [384,640]
img0_shape = (1080, 1920)

oncpu = tensor(coords)
outc = scale_coords(img1_shape, oncpu[:,:4], img0_shape)
print(f'on cpu {outc}')

ongpu = tensor(coords, device=device )
outg = scale_coords(img1_shape, ongpu[:,:4] , img0_shape) 
print(f'on gpu {outg}')

import torch
    
print (torch.__version__)

for ii in range(10000):
    fa = torch.ones(499, 526, device='mps')

    #print (fa[0][0]) # Fine
    print (fa[100][100]) # Segfault after a random number of iterations