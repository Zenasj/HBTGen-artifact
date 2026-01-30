import torch.nn as nn
import math

torch.onnx.export(model,       
     dummy_input,      
     "TextBPM_dyn.onnx",       
     export_params=True, 
     opset_version=16,    
     do_constant_folding=True,  
     input_names = ['modelInput'],   
     output_names = ['modelOutput'], 
     dynamic_axes={'modelInput' : [0, 2, 3],    
                            'modelOutput' : {0 : 'batch_size'}})

#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function

import sys
import numpy as np
import torch


def pad_center(data, *, size, axis = -1, **kwargs):
    
    kwargs.setdefault("mode", "constant")

    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.dim()
    lengths[axis] = (lpad, int(size - n - lpad))

    if lpad < 0:
        raise ValueError(f"Target size ({size:d}) must be at least input size ({n:d})")

    # Convert the list of tuples to a tuple of tuples
    pad_tuple = ()
    for l in lengths:
        for li in l:
            pad_tuple += (li,)

    return torch.nn.functional.pad(data, pad_tuple, **kwargs)



class STFT(torch.nn.Module):
    def __init__(self, win_length, hop_length, nfft, window, window_periodic = True, stft_mode = None):
        super().__init__()
        self.win_length = win_length
        self.hop_length = hop_length
        self.nfft = nfft #2**math.ceil(math.log2(self.win_length))
        self.freq_cutoff = self.nfft // 2 + 1
        self.window = window

        if stft_mode == 'conv':
            fourier_basis = torch.view_as_real(torch.fft.fft2(torch.eye(self.nfft), dim = 1))
            forward_basis = fourier_basis[:self.freq_cutoff].permute(2, 0, 1).reshape(-1, 1, fourier_basis.shape[1])
            forward_basis = forward_basis * pad_center(self.window, size = self.nfft)
            self.stft = torch.nn.Conv1d(
                forward_basis.shape[1],
                forward_basis.shape[0],
                forward_basis.shape[2],
                bias = False,
                stride = self.hop_length
            ).requires_grad_(False)
            self.stft.weight.copy_(forward_basis)
        else:
            self.stft = None

    def forward(self, signal):
        pad = self.freq_cutoff - 1        
        padded_signal = torch.nn.functional.pad(signal.unsqueeze(1), (pad, pad), mode = 'reflect').squeeze(1)
        real, imag = self.stft(padded_signal.unsqueeze(dim = 1)).split(self.freq_cutoff, dim = 1) if self.stft is not None else padded_signal.stft(self.nfft, hop_length = self.hop_length, win_length = self.win_length, window = self.window, center = False).unbind(dim = -1)
        realimag = torch.stack((real, imag), dim=3)

        return realimag


class Model(torch.nn.Module):
    """ Model definition
    """
    def __init__(self):
        super(Model, self).__init__()
       
        self.window = torch.hamming_window(30)
        self.stft = STFT(30, 160, 1024, self.window, window_periodic = True, stft_mode = 'conv')

        self.max_dim=2
        
        self.bn=torch.nn.BatchNorm2d(1, affine=False)
       
        
        return
    
    def forward(self, x):
        x=x.squeeze(-1)
        x = self.stft(x)

        x, _ = x.max(self.max_dim)
        x=x.unsqueeze(1)

        score = self.bn(x)

        return score

import importlib
import torch
import numpy as np

module_model="model.py" #defined above
dim=None
device='cpu'

m = importlib.import_module("model") #load model defined above
model = m.Model()
model.to(device, dtype=torch.float32)
model.eval()

length=60000
data = np.zeros([length, 1], dtype=np.float32)
data=torch.as_tensor(data)
data=torch.unsqueeze(data,0)
data = data.to(device, dtype=torch.float32)

#This works OK
scores = model.forward(data)


input_names = ["pcm"]
output_names = ["score"]

torch.onnx.export(model, data, "model.onnx", verbose=True, input_names=input_names, output_names=output_names, dynamic_axes={'pcm' : {1 : 'time'},}) #This fails