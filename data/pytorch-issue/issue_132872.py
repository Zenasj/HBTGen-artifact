import torch
import torch.nn as nn

add_3: "f32[64, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_3, _frozen_param195)

add_3: "f32[64, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_3, mul_4);

B, N, C = x.shape
pos_score = self.rel_indices.expand(B, -1, -1, -1)
pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score

self.pos_proj = nn.Linear(3, num_heads)
self.rel_indices: torch.Tensor = torch.zeros(1, 1, 1, 3)
# if we use self.register_buffer("rel_indices", torch.zeros(1, 1, 1, 3), False) in bad commit, the regression will disppear

def forward(self, x):
        B, N, C = x.shape
        if self.rel_indices is None or self.rel_indices.shape[1] != N:
            self.rel_indices = self.get_rel_indices(N)

def get_rel_indices(self, num_patches: int) -> torch.Tensor:
        img_size = int(num_patches ** .5)
        rel_indices = torch.zeros(1, num_patches, num_patches, 3)
        ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size, img_size)
        indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
        indd = indx ** 2 + indy ** 2
        rel_indices[:, :, :, 2] = indd.unsqueeze(0)
        rel_indices[:, :, :, 1] = indy.unsqueeze(0)
        rel_indices[:, :, :, 0] = indx.unsqueeze(0)
        device = self.qk.weight.device
        return rel_indices.to(device)

mkldnn:_conv_pointwise