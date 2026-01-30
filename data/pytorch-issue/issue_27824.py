import torch.nn.functional as F

import torch
# import matplotlib.pyplot as plt


# TESTING GLOBALS
FLOAT_DTYPE = torch.float32  # also reproducible with float64
MASK_DTYPE = torch.uint8 # torch.bool
H, W = 400, 640  # also reproducible with many different resolutions
DEVICE = "cpu"  # reproducible on i7 cpu and RTX2060 cuda, Ubuntu17.10
SOME_FLOAT = 1.23
MIN_IDX = 1  # high values cause segfault

# CREATE AND INITIALIZE BUFFERS
D_TMP = torch.empty(H, W, dtype=FLOAT_DTYPE).to(DEVICE).fill_(SOME_FLOAT)
MASK_BUFF = torch.ones(H, W, dtype=MASK_DTYPE).to(DEVICE)

# CRITICAL PART
#
# D_TMP *= MASK_BUFF  # OK
# D_TMP[0:H, 0:W] *= MASK_BUFF[0:H, 0:W]  # OK
# D_TMP[[MIN_IDX:H, 0:W] *= MASK_BUFF[[MIN_IDX:H, 0:W]  # OK
D_TMP[MIN_IDX:H, MIN_IDX:W] *= MASK_BUFF[MIN_IDX:H, MIN_IDX:W]  # buggy
# D_TMP[0:H, MIN_IDX:W] *= MASK_BUFF[0:H, MIN_IDX:W]  # buggy


# CHECK IF ENTRIES ARE MISSING
assert MASK_BUFF.all(), "Mask has zeros?"
assert (D_TMP==SOME_FLOAT).all(), "D_tmp has dropped values?"
# plt.imshow(MASK_BUFF.to(torch.uint8).cpu().numpy())
# plt.imshow(D_TMP.cpu().numpy())
# plt.show()