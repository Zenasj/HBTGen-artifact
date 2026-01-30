import torch.nn as nn
import numpy as np

att = nn.MultiheadAttention(np.int64(5), 1, kdim=np.int64(2),vdim=np.int64(2))

self._qkv_same_embed_dim

False

self._qkv_same_embed_dim

bool