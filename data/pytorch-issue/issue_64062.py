py
import torch 
import os
import urllib.request
import shutil

inception_url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt"

if not os.path.exists('/tmp/inception-2015-12-05.pt'):
    # download the file
    with urllib.request.urlopen(inception_url) as response, open(inception_path, 'wb') as f:
        shutil.copyfileobj(response, f)

net = torch.jit.load('/tmp/inception-2015-12-05.pt').eval().cuda()

x = torch.randn(1, 3, 299, 299, device='cuda')

print('PyTorch version', torch.__version__)
print('full net(x)', net(x).sum())
torch._C._jit_override_can_fuse_on_gpu(False)
print('net.layers(x) w/o fuser', net.layers(x).sum())
torch._C._jit_override_can_fuse_on_gpu(True)
print('net.layers(x) w/ fuser', net.layers(x).sum())