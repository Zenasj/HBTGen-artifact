import torch.nn as nn

from math import inf
import torch
from torch import tensor, device
import torch.fx as fx
import torchdynamo
from torchdynamo.testing import rand_strided
from torchdynamo.debug_utils import run_fwd_maybe_bwd
from torchdynamo.debug_utils import same_two_models

args = [((2, 3, 224, 224), (150528, 50176, 224, 1), torch.float32, 'cuda', False)]
args = [rand_strided(sh, st, dt, dev).requires_grad_(rg) for (sh, st, dt, dev, rg) in args]


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_self_stem_proj = Conv2d(3, 384, kernel_size=(16, 16), stride=(16, 16))
        self.self_self_stem_norm = Identity()
        self.self_self_blocks_0__norm1 = LayerNorm((384,), eps=1e-06, elementwise_affine=True)
        self.self_self_blocks_0__mlp_tokens_fc1 = Linear(in_features=196, out_features=384, bias=True)
        self.self_self_blocks_0__mlp_tokens_act = SiLU()
        self.self_self_blocks_0__mlp_tokens_drop1 = Dropout(p=0.0, inplace=False)
        self.self_self_blocks_0__mlp_tokens_fc2 = Linear(in_features=192, out_features=196, bias=True)
        self.self_self_blocks_0__mlp_tokens_drop2 = Dropout(p=0.0, inplace=False)
        self.self_self_blocks_0__drop_path = Identity()
        self.self_self_blocks_0__norm2 = LayerNorm((384,), eps=1e-06, elementwise_affine=True)
        self.self_self_blocks_0__mlp_channels_fc1 = Linear(in_features=384, out_features=1536, bias=True)
        self.self_self_blocks_0__mlp_channels_act = SiLU()
        self.self_self_blocks_0__mlp_channels_drop1 = Dropout(p=0.0, inplace=False)
        self.self_self_blocks_0__mlp_channels_fc2 = Linear(in_features=768, out_features=384, bias=True)
        self.self_self_blocks_0__mlp_channels_drop2 = Dropout(p=0.0, inplace=False)
        self.self_self_norm = LayerNorm((384,), eps=1e-06, elementwise_affine=True)
        self.self_self_head = Linear(in_features=384, out_features=1000, bias=True)



    def forward(self, x : torch.Tensor):
        self_self_stem_proj = self.self_self_stem_proj(x);  x = None
        flatten = self_self_stem_proj.flatten(2);  self_self_stem_proj = None
        transpose = flatten.transpose(1, 2);  flatten = None
        self_self_stem_norm = self.self_self_stem_norm(transpose);  transpose = None
        self_self_blocks_0__norm1 = self.self_self_blocks_0__norm1(self_self_stem_norm)
        transpose_1 = self_self_blocks_0__norm1.transpose(1, 2);  self_self_blocks_0__norm1 = None
        self_self_blocks_0__mlp_tokens_fc1 = self.self_self_blocks_0__mlp_tokens_fc1(transpose_1);  transpose_1 = None
        chunk = self_self_blocks_0__mlp_tokens_fc1.chunk(2, dim = -1);  self_self_blocks_0__mlp_tokens_fc1 = None
        getitem = chunk[0]
        getitem_1 = chunk[1];  chunk = None
        self_self_blocks_0__mlp_tokens_act = self.self_self_blocks_0__mlp_tokens_act(getitem_1);  getitem_1 = None
        mul = getitem * self_self_blocks_0__mlp_tokens_act;  getitem = self_self_blocks_0__mlp_tokens_act = None
        self_self_blocks_0__mlp_tokens_drop1 = self.self_self_blocks_0__mlp_tokens_drop1(mul);  mul = None
        self_self_blocks_0__mlp_tokens_fc2 = self.self_self_blocks_0__mlp_tokens_fc2(self_self_blocks_0__mlp_tokens_drop1);  self_self_blocks_0__mlp_tokens_drop1 = None
        self_self_blocks_0__mlp_tokens_drop2 = self.self_self_blocks_0__mlp_tokens_drop2(self_self_blocks_0__mlp_tokens_fc2);  self_self_blocks_0__mlp_tokens_fc2 = None
        transpose_2 = self_self_blocks_0__mlp_tokens_drop2.transpose(1, 2);  self_self_blocks_0__mlp_tokens_drop2 = None
        self_self_blocks_0__drop_path = self.self_self_blocks_0__drop_path(transpose_2);  transpose_2 = None
        add = self_self_stem_norm + self_self_blocks_0__drop_path;  self_self_stem_norm = self_self_blocks_0__drop_path = None
        self_self_blocks_0__norm2 = self.self_self_blocks_0__norm2(add)
        self_self_blocks_0__mlp_channels_fc1 = self.self_self_blocks_0__mlp_channels_fc1(self_self_blocks_0__norm2);  self_self_blocks_0__norm2 = None
        chunk_1 = self_self_blocks_0__mlp_channels_fc1.chunk(2, dim = -1);  self_self_blocks_0__mlp_channels_fc1 = None
        getitem_2 = chunk_1[0]
        getitem_3 = chunk_1[1];  chunk_1 = None
        self_self_blocks_0__mlp_channels_act = self.self_self_blocks_0__mlp_channels_act(getitem_3);  getitem_3 = None
        mul_1 = getitem_2 * self_self_blocks_0__mlp_channels_act;  getitem_2 = self_self_blocks_0__mlp_channels_act = None
        self_self_blocks_0__mlp_channels_drop1 = self.self_self_blocks_0__mlp_channels_drop1(mul_1);  mul_1 = None
        self_self_blocks_0__mlp_channels_fc2 = self.self_self_blocks_0__mlp_channels_fc2(self_self_blocks_0__mlp_channels_drop1);  self_self_blocks_0__mlp_channels_drop1 = None
        self_self_blocks_0__mlp_channels_drop2 = self.self_self_blocks_0__mlp_channels_drop2(self_self_blocks_0__mlp_channels_fc2);  self_self_blocks_0__mlp_channels_fc2 = None
        self_self_blocks_0__drop_path_1 = self.self_self_blocks_0__drop_path(self_self_blocks_0__mlp_channels_drop2);  self_self_blocks_0__mlp_channels_drop2 = None
        add_1 = add + self_self_blocks_0__drop_path_1;  add = self_self_blocks_0__drop_path_1 = None
        self_self_norm = self.self_self_norm(add_1);  add_1 = None
        mean = self_self_norm.mean(dim = 1);  self_self_norm = None
        self_self_head = self.self_self_head(mean);  mean = None
        return (self_self_head,)



mod = Repro().cuda()
opt_mod = torchdynamo.optimize("inductor")(mod)


mod.eval()
opt_mod.eval()
with torch.cuda.amp.autocast(enabled=False):
    assert same_two_models(mod, mod, args), "Eager itself failed"
    assert same_two_models(mod, opt_mod, args), "Dynamo failed"