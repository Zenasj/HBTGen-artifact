import torch.nn as nn

import numpy as np
import torch
img = np.zeros((2048,2048),dtype=np.float32)
img[500,300] = 1.0
psf_half = 25
conv_size = 2*psf_half+1
psf = np.ones((conv_size,conv_size), dtype = np.float32)

weights_t = torch.tensor(psf[np.newaxis,np.newaxis,:,:]) #Numpy to tensor
img_t = torch.tensor(img[np.newaxis,np.newaxis,:,:]) # batch size 1, 1 channel -- 1 color
img_conv_pytorch = torch.nn.functional.conv2d(img_t, weights_t, padding=psf_half)

import torch.backends.mkldnn
with torch.backends.mkldnn.flags(enabled=False):
    img_conv_pytorch = torch.nn.functional.conv2d(img_t, weights_t, padding=psf_half)