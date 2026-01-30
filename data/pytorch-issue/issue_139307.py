import torch.nn as nn

import torch

class Model(torch.nn.Module):
    def forward(self, x):
        # torch._check(2 <= x.shape[0] <= 3)  # this works
        torch._check(x.shape[0] in {2, 3})
        return x

torch.export.export(
    Model(),
    (torch.zeros(2),),
    dynamic_shapes={
        "x": {
            0: torch.export.Dim("discrete", min=2, max=3)
        }
    }
)

import torch

class ImgVideoModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_dim = 8
        self.vid_len = 12
        self.img_pos_embed = torch.nn.Parameter(torch.zeros(1, 1, self.emb_dim))
        self.vis_pos_embed = torch.nn.Parameter(torch.zeros(1, self.vid_len, self.emb_dim))
        self.projector = TransformerWithLargeParameters(...)
    def forward(self, x):
        # x: [B, 1, C] or [B, 12, C]
        B, T, C = x.size()
        x = torch.cond(T == 1, lambda t: t + self.img_pos_embed, lambda t: t + self.vis_pos_embed, (x,))
        # This is the intended behavior of the model.
        # if T == 1:
        #     x = x + self.img_pos_embed
        # else:
        #     assert T == 12
        #     x = x + self.vis_pos_embed
        x = self.projector(x)
        return x

inp = torch.rand(2, 12, 8)
bs = torch.export.Dim("bs")
seq_len = torch.export.Dim("seq_len", min=1, max=12)
ep = torch.export.export(ImgVideoModel(), (inp,), dynamic_shapes={"x": {0: bs, 1: seq_len}})
print(ep)

import torch

class Model(torch.nn.Module):
    def forward(self, x):
        torch._check(x.shape[0] in {2, 3})
        return x

torch.export.export(
    Model(),
    (torch.zeros(3),),
    dynamic_shapes={
        "x": {
            0: torch.export.Dim("x", min=2, max=3)
        }
    }
)

torch.export.Dim("x", values={2, 3})

torch._check(x.shape[0] == 2) or torch._check(x.shape[0] == 3)