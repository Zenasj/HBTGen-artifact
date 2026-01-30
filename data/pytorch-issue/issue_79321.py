import numpy as np

class SubClassedTensor(torch.Tensor): 
    pass

if config.subclass:
    images = images.as_subclass(SubClassedTensor)

if config.subclass:
    images = images.as_subclass(SubClassedTensor)

from fastai.vision.all import *

imagenette = untar_data(URLs.IMAGENETTE_320)

dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                    splitter=GrandparentSplitter(valid_name='val'),
                    get_items=get_image_files, get_y=parent_label,
                    item_tfms=Resize(256),
                    batch_tfms=Normalize.from_stats(*imagenet_stats))
dls =  dblock.dataloaders(imagenette, bs=64, num_workers=num_cpus(), pin_memory=True)

learn = Learner(dls, resnet50(num_classes=dls.c)).to_fp16()
learn.remove_cb(CastToTensor) # Comment out this line to train with torch.Tensor
learn.fit_one_cycle(3, 3e-3)

import torch
from torch.utils.benchmark import Timer
import torchvision


init = """
from torchvision.models import resnet50

dtype = torch.float16
device = "cuda"
use_subclass = False
run_bw = False

m = resnet50()
m.to(dtype=dtype, device=device)

class SubClassedTensor(torch.Tensor): 
    pass

inp = torch.rand(32, 3, 240, 240, dtype=dtype, device=device)
if use_subclass:
    inp = inp.as_subclass(SubClassedTensor)
"""

stmt = """
out = m(inp)
if run_bw:
    out.sum().backward()
"""

t = Timer(stmt, init).blocked_autorange()
print(t)