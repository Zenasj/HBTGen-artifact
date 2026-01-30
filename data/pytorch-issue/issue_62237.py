import torch.nn as nn

import torch

s = 10
t_in = torch.arange(s, dtype=torch.float).unsqueeze(0).unsqueeze(0)
scale = 1.00001

output_size = (1, 1, s)
resized = torch.zeros(output_size)
scales = (1, 1, 1.0 / scale)

for i in range(resized.shape[2]):
    x = i * scales[2]  # <--- this buggy and will be fixed soon by x = (i + 0.5) * scales[2] - 0.5
    ii = int(x)
    resized[0, 0, i] = t_in[0, 0, ii]

print(resized)

scale = 1.0 * isize / osize

for o in range(osize):
    x = (o + 0.5) * scale - 0.5  # vs o * scale
    i = round(x)  # vs floor(x)
    output[o] = input[i]

import torch
import numpy as np
import torch.nn.functional as F

x = np.array([[[
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
]]])
y = F.interpolate(
    torch.from_numpy(x.astype(np.float32)), **{
        'size': (3, 1),
        'scale_factor': None,
        'mode': 'bilinear',
        'align_corners': None,
        'recompute_scale_factor': False
    })
print(f"torch.__version__: {torch.__version__}\n" f"{y}")

import torch
import numpy as np
import torch.nn.functional as F
import cv2
import numpy as np


osize = (12, 12)
isize = (32, 32)

x = np.arange(isize[0] * isize[1]).reshape(1, 1, *isize)

pth_res = F.interpolate(
    torch.from_numpy(x.astype(np.float32)), **{
        'size': osize,
        'scale_factor': None,
        'mode': 'bilinear',
        'align_corners': None,
        'recompute_scale_factor': False
    })
cv_res = cv2.resize(x[0, 0, ...].astype("float"), dsize=osize[::-1], interpolation=cv2.INTER_LINEAR)

print(f"torch.__version__: {torch.__version__}\n")
print("pth: ", pth_res[0, 0, :5, :5].numpy())
print("opencv: ", cv_res[:5, :5])

assert np.allclose(pth_res[0, 0, ...].numpy(), cv_res)