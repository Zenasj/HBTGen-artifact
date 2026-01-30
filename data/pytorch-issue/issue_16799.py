import torch.nn as nn

import numpy as np
import torch
import torch.utils.model_zoo as model_zoo
import torch.onnx
from torch import nn
from collections import OrderedDict
from model import G_mapping, G_synthesis, Truncation

def stylegan_G(batch_size=1):
    '''
    convert styleGan generate network (synthesis part only)
    '''
    stylegan_G_onnx_model = "onnx/" + "stylegan_G.onnx"
    weight_path = "./karras2019stylegan-ffhq-1024x1024.for_g_all.pt"
    g_all = nn.Sequential(OrderedDict([
        ('g_mapping', G_mapping()),
        ('g_synthesis', G_synthesis())    
    ]))
    g_all.load_state_dict(torch.load(weight_path))

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    g_all.eval()
    g_all.to(device)
    x = torch.randn(batch_size, 18, 512).cuda()
    torch_out = torch.onnx._export(g_all[1],
                                x,
                                stylegan_G_onnx_model,
                                export_params=True)
    return torch_out, x

stylegan_G()