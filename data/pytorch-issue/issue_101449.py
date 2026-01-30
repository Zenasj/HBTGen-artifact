# RuntimeError: CUDA error: an illegal memory access was encountered
# happens on iteration 3 if fused=True on the Adam optimizer
# Happens with 3B or 4B parameters, but not 1B.
#
# Tested on A100 80GB
#
# Nvidia Pytorch container 23.04-py3
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
#
# torch version:  2.1.0a0+fe05266
# cuda version :  12.1
# cudnn version:  8900

use_fused = True

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseNet(nn.Module):
    def __init__(self, in_dim, out_dim, max_parameters, increment, final_bias=True):
        super().__init__()        
        total_parameters = 0
        dim = in_dim
        self.linear_layers = nn.ParameterList()
        bias = False
        while total_parameters < max_parameters:
            if total_parameters + dim * increment + (dim + increment) * out_dim >= max_parameters:
                break
            self.linear_layers.append(nn.Linear(dim, increment, bias=bias) )
            total_parameters += dim * increment
            dim += increment
        self.num_acts = dim
        self.final_layer = nn.Linear(dim, out_dim, bias=final_bias)
        
    def forward(self, x):
        input = x
        for layer in self.linear_layers:
            x2 = layer(x)
            x2 = F.relu(x2)
            x = torch.cat( (x,x2), dim=1 )                
        x = self.final_layer(x)        
        return x

dev = 'cuda'    
frame_width = 64
frame_height = 32
image_channels = 3
frame_size = frame_width*frame_height
observation_size = 256
fovea = 16    
batch = 1

print('torch version: ', torch.__version__)
print('cuda version : ', torch.version.cuda)
print('cudnn version: ', torch.backends.cudnn.version())

print('building model')
reconstruct_parms = 3_000_000_000
reconstruct_step = 256
weights_model = DenseNet(2, observation_size * frame_size, reconstruct_parms, reconstruct_step, final_bias=False).to(device=dev)
print(weights_model)

optim = torch.optim.Adam(weights_model.parameters(), fused=use_fused)

fovea_flat = torch.zeros(batch, 3, observation_size, device=dev)
fovea_xy = torch.zeros(batch,2, device=dev)
frames_gpu_flat = torch.zeros(batch, frame_size*image_channels, device=dev)

for u in range(100):            
    generated_weights = weights_model(fovea_xy)
    generated_weights = generated_weights.view(batch, observation_size, frame_size)
    reconstruction = fovea_flat @ generated_weights
    reconstruction = reconstruction.permute(0,2,1).reshape(batch, frame_size*image_channels)
    decode_loss = F.mse_loss(reconstruction, frames_gpu_flat)

    print(u)
    optim.zero_grad()
    decode_loss.backward()        
    optim.step()
        
print('done')