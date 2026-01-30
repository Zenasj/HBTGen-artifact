import torch.nn as nn

import math
from typing import List

import torch
from torch import nn
import torch.nn.functional as F


# Some of the following codes come from padding.py in rwightman/pytorch-image-models repo.
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/padding.py
def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)

class TestModule(nn.Module):
    def __init__(self, k, s):
        super(TestModule, self).__init__()
        self.k = [k, k]
        self.s = [s, s]
        self.d = [1, 1]
        self.value = 0.0

    def forward(self, x):
        ih, iw = x.size()[-2:]
        pad_h, pad_w = get_same_padding(ih, self.k[0], self.s[0], self.d[0]), get_same_padding(iw, self.k[1], self.s[1], self.d[1])
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=self.value)
        return x

class Wrapper(nn.Module):
    def __init__(self, k, s):
        super(Wrapper, self).__init__()
        self.first = TestModule(k, s)
        self.second = TestModule(k, s)

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)  # <- if commented out, no error.
        return x


def main():
    model = Wrapper(k=3, s=1)
    model.eval()
    model = torch.jit.script(model)

    x = torch.randn((1, 3, 150, 150))
    example_outputs = model(x)
    print(example_outputs)

    torch.onnx.export(
        model,
        x,
        './tmp/minimum.onnx',
        verbose=True,
        input_names=['x'],
        output_names=['out'],
        dynamic_axes=None,
        opset_version=11,
        example_outputs=example_outputs)


if __name__ == '__main__':
    main()

"""
This code is based on validate.py in original effdet.
https://github.com/rwightman/efficientdet-pytorch/blob/master/validate.py
"""

import sys
import torch
from timm.utils import setup_default_logging
from timm.models.layers import set_layer_config

# Manually adding a path to efficientdet-pytorch
sys.path.append('./efficientdet-pytorch/')
from effdet import create_model
from effdet.config import model_config


def main():
    modelname = 'tf_efficientdet_d7x'
    exportpath = 'repro.onnx'
    torchscript = True
    no_jit = True
    exportable = True

    num_classes = None
    redundant_bias = None
    soft_nms = None
    checkpoint = ''
    use_ema = False

    # create model
    with set_layer_config(scriptable=torchscript, exportable=exportable, no_jit=no_jit):
        model = create_model(
            modelname,
            bench_task='',
            num_classes=num_classes,
            pretrained=True,
            redundant_bias=redundant_bias,
            soft_nms=soft_nms,
            checkpoint_path=checkpoint,
            checkpoint_ema=use_ema,
        )
        model.eval()
        image_size = model.config['image_size']

        dummy_input = torch.randn(1, 3, image_size[0], image_size[1])

        print('Apply torch.jit.script()')
        model = torch.jit.script(model)
        example_outputs = model(dummy_input)

        # export model.
        input_names = ['images']
        output_names = ['classes', 'boxes']
        dynamic_axes = None
        torch.onnx.export(
            model,
            dummy_input,
            exportpath,
            verbose=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=11,
            example_outputs=example_outputs)
    print('done.')


if __name__ == '__main__':
    main()