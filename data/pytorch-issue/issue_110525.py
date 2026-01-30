import torch

from torch.nn import Transformer 
from torch import vmap, rand

transformer = Transformer()
vmap_transformer = vmap(transformer, in_dims=(1, None), randomness="same")
src, tgt = rand((32, 8, 512)), rand((32, 512))
out = vmap_transformer(src, tgt)