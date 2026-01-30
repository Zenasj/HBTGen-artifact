import torch.nn as nn

"""
conda install python=3.10 cudatoolkit=11.7 -c nvidia
python -m pip install numpy ninja --pre torch[dynamo] torchvision --force-reinstall --extra-index-url https://download.pytorch.org/whl/nightly/cu117
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
# (this can take dozens of minutes)
"""
import torch
import torch._dynamo as dynamo
import xformers.ops as xops

torch._dynamo.config.verbose=True

device = "cuda"
dtype = torch.half

class xFormersMHA(torch.nn.Module):
    def forward(self, q, k, v):
        return xops.memory_efficient_attention(q, k, v)
model = xFormersMHA().to(device).to(dtype)
inp = torch.zeros([2, 10, 16, 128]).to(device).to(dtype)

print("torch version", torch.__version__)
model = dynamo.optimize()(model)

print("Running forward")
model(inp, inp, inp)