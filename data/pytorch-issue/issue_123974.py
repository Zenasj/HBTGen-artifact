'''
Install:
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
pip3 install monai
'''

import torch
from monai.networks.nets import SegResNet

model = SegResNet(
    blocks_down=(1, 2, 2, 4),
    blocks_up=(1, 1, 1),
    init_filters=16,
    in_channels=4,
    out_channels=3,
    dropout_prob=0.2
).to('cuda')
data = torch.randn(1, 4, 224, 224, 128).cuda()
torch.export.export(model, args=(data,))