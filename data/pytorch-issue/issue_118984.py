import torch.nn as nn

#minified2.py

import torch
from torch import nn
import os
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch._dynamo
#torch._dynamo.config.optimize_ddp = False
device = torch.device(f'cuda:{0}')

rank = os.environ.get('LOCAL_RANK','-1')
if int(rank) >=0:
    device = torch.device(f'cuda:{rank}')
    print(device)
    torch.cuda.set_device(device)
    dist.init_process_group(backend='nccl', init_method='env://')

class nn_Conv2d(nn.Conv2d):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def forward(self,x):
        if not x.is_contiguous() and self.kernel_size[0]==self.kernel_size[1]==1 and self.stride[0]==self.stride[1]==1 \
                and self.padding[0]==self.padding[1]==0:
            x=x.permute(0,2,3,1)
            x=nn.functional.linear(x,self.weight.flatten(1),self.bias.flatten() if self.bias is not None else None)
            x=x.permute(0,3,1,2)
            return x
        else:
            return super().forward(x)

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class LayerNorm(nn.Module):
    def __init__(self,shape,affine=True,use_bias=True,elementwise_affine=None,
                 dim=1, eps=1e-5):
        super().__init__()
        if elementwise_affine is not None:
            affine=use_bias=elementwise_affine
        self.eps=eps
        self.shape=shape
        self.affine=affine
        self.use_bias=use_bias
        self.dim=dim
        if affine:
            self.weight = nn.Parameter(torch.ones(shape))
            nn.init.ones_(self.weight)
        else:
            self.weight = None
        if use_bias:
            self.bias = nn.Parameter(torch.ones(shape))
            nn.init.torch.nn.init.normal_(self.bias, 0.0003)
        else:
            self.bias=None

    def forward(self,x):
        if getattr(self,'dim',1)==1:
            dim=x.dim()
            x=x.permute(0,2,3,1) if dim == 4 else x.permute(0,2,1)
            x=nn.functional.layer_norm(x,[self.shape],self.weight,self.bias,self.eps)
            x=x.permute(0,3,1,2) if dim == 4 else x.permute(0,2,1)
            return x

        view_shape=[1] * len(x.shape)
        view_shape[getattr(self,'dim',1)]=self.shape
        if getattr(self,'use_bias',True):
            u=x.mean(getattr(self,'dim',1),keepdim=True)
            s = (x-u).pow(2).mean(getattr(self,'dim',1),keepdim=True)
            x=(x-u)/torch.sqrt(s+self.eps)
        else:
            s = x.pow(2).mean(getattr(self,'dim',1),keepdim=True)
            x=x/torch.sqrt(s+self.eps)

        if getattr(self,'affine',True):
            x=self.weight.view(view_shape) * x
        if getattr(self,'use_bias',True):
            x= x+ self.bias.view(view_shape)
        return x

class ConvL(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn_Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.norm = LayerNorm(c2)
        self.act = nn.GELU() if act else nn.Identity()

    @torch.compile
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class BottleneckL(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvL(c1, c_, 1, 1)
        self.cv2 = ConvL(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    @torch.compile
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

from torch.utils.checkpoint import checkpoint
class BottleneckCSP2L(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2)  # hidden channels
        self.cv1 = ConvL(c1, c_, 1, 1)
        self.cv2 = nn_Conv2d(c_, c_, 1, 1, bias=False)
        self.cv3 = ConvL(2 * c_, c2, 1, 1)
        self.norm = LayerNorm(2 * c_)
        self.act = nn.GELU()
        self.m = nn.Sequential(*[BottleneckL(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1) if not self.training else checkpoint(self.m,x1, use_reentrant=True)
        y2 = self.cv2(x1)
        return self.cv3(self.act(self.norm(torch.cat((y1, y2), dim=1))))



model = nn.Sequential(
                      ConvL(64,256,1,1),
                      BottleneckCSP2L(256,256,3),
                      )
model = model.cuda()
model.train()
print(model)

x = torch.rand(4,64,80,80).cuda()
x.requires_grad=True
if int(rank) >=0:
    model = DDP(model.cuda(),device_ids=[int(rank)], output_device=int(rank))
optimizer = torch.optim.AdamW(model.parameters())

with torch.cuda.amp.autocast():
    y=model(x)
y[0].sum().backward()

optimizer.step()
optimizer.zero_grad()
torch.cuda.synchronize()
print('test 1 done')

x = torch.rand(2,64,96,96).cuda()
x.requires_grad=True

with torch.cuda.amp.autocast():
    y=model(x)
y[0].sum().backward()

optimizer.step()
optimizer.zero_grad()
torch.cuda.synchronize()