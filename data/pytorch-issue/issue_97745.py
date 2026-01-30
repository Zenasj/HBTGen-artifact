import torch.nn as nn

import torch
import torch._dynamo
from torch._dynamo.debug_utils import run_fwd_maybe_bwd
import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
from torch.nn import *

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).cuda()
    def forward(self, inputs : torch.Tensor):
        return self.conv(inputs)

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = Model()
        self.discriminator = Model()
    def d_step(self, batch_outputs: torch.Tensor):
        fake_d_pred = self.discriminator(batch_outputs.detach())
        d_loss = fake_d_pred.mean()
        d_loss.backward()
    def forward(self, inputs : torch.Tensor):
        batch_inputs = torch.rand((1,3,32,32)).cuda()
        batch_gt_data =  torch.rand((1,3,32,32)).cuda()
        batch_outputs = self.generator(batch_inputs)
        set_requires_grad(self.discriminator, False)
        g_loss = abs(batch_outputs-batch_gt_data).mean()
        g_loss.backward()
        set_requires_grad(self.discriminator, True)
        self.d_step(batch_outputs)

mod = Repro()
opt_mod = torch._dynamo.optimize("aot_eager")(mod)

with torch.cuda.amp.autocast(enabled=False):
    ref = run_fwd_maybe_bwd(mod, (torch.rand((1,3,32,32)).cuda()))

import torch
import torch._dynamo
from torch._dynamo.debug_utils import run_fwd_maybe_bwd
import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
from torch.nn import *

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).cuda()
    def forward(self, inputs : torch.Tensor):
        return self.conv(inputs)

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = Model()
        self.discriminator = Model()
    def d_step(self, batch_outputs: torch.Tensor):
        fake_d_pred = self.discriminator(batch_outputs.detach())
        d_loss = fake_d_pred.mean()
        d_loss.backward()
    def forward(self, inputs : torch.Tensor):
        batch_inputs = torch.rand((1,3,32,32)).cuda()
        batch_gt_data =  torch.rand((1,3,32,32)).cuda()
        batch_outputs = self.generator(batch_inputs)
        set_requires_grad(self.discriminator, False)
        g_loss = abs(batch_outputs-batch_gt_data).mean()
        g_loss.backward()
        set_requires_grad(self.discriminator, True)
        self.d_step(batch_outputs)

mod = Repro()
opt_mod = torch._dynamo.optimize("aot_eager")(mod)

with torch.cuda.amp.autocast(enabled=False):
    # ref = run_fwd_maybe_bwd(mod, (torch.rand((1,3,32,32)).cuda()))
    res = run_fwd_maybe_bwd(opt_mod, (torch.rand((1,3,32,32)).cuda()))

import torch
import torch._dynamo
from torch._dynamo.debug_utils import run_fwd_maybe_bwd
import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
from torch.nn import *

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).cuda()
    def forward(self, inputs : torch.Tensor):
        return self.conv(inputs)

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = Model()
        self.discriminator = Model()
    def d_step(self, batch_outputs: torch.Tensor):
        fake_d_pred = self.discriminator(batch_outputs.detach())
        d_loss = fake_d_pred.mean()
        d_loss.backward()
    def forward(self, inputs : torch.Tensor):
        batch_inputs = torch.rand((1,3,32,32)).cuda()
        batch_gt_data =  torch.rand((1,3,32,32)).cuda()
        batch_outputs = self.generator(batch_inputs)
        set_requires_grad(self.discriminator, False)
        g_loss = abs(batch_outputs-batch_gt_data).mean()
        g_loss.backward()
        set_requires_grad(self.discriminator, True)
        self.d_step(batch_outputs.detach()) ################# use `detach()`, before function calls

mod = Repro()
opt_mod = torch._dynamo.optimize("aot_eager")(mod)

with torch.cuda.amp.autocast(enabled=False):
    # ref = run_fwd_maybe_bwd(mod, (torch.rand((1,3,32,32)).cuda()))
    res = run_fwd_maybe_bwd(opt_mod, (torch.rand((1,3,32,32)).cuda()))