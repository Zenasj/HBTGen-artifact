quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

layer = nn.TransformerEncoderLayer(d_model=size,
                                   nhead=8,
                                   dim_feedforward=size * decoder_girth,
                                   dropout=dropout)
self.decoder = nn.TransformerEncoder(layer, decoder_layers)

quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

def forward(self, x):
        if self.denoise:
            raise NotImplementedError()
        else:
            if self.decoder_type == 'transformer':
                encoded = self.layers(x)  # this is my encoder, CNNs, I do NOT quantize them here
                # https://pytorch.org/docs/stable/nn.html#transformer
                # src: (S, N, E)
                # instead of  batch  * channels * length
                return self.decoder(
                    encoded.permute(2, 0, 1).contiguous()
                    ).permute(1, 2, 0).contiguous()  ## error happens here!
            else:
                raise NotImplementedError('Forward function for {} decoder not implemented'.format(self.decoder))

import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, dim=512, n_heads=2, has_out_proj=True, single_matrix=True):
        super().__init__()
        assert dim % n_heads == 0

        if single_matrix:
            # keep order in accordance with PyTorch implementation
            # https://github.com/zhangguanheng66/pytorch/blob/6c743c7721251ca9b5046fc56a071bc1f36916be/torch/nn/functional.py#L3182
            self.QKV = nn.Linear(dim, 3 * dim)
        else:
            self.K = nn.Linear(dim, dim)
            self.Q = nn.Linear(dim, dim)
            self.V = nn.Linear(dim, dim)

        self.single_matrix = single_matrix
        self.scale = (dim / n_heads) ** 0.5
        self.n_heads = n_heads
        self.has_out_proj = has_out_proj
        if self.has_out_proj:
            self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        bsz, seq, dim = x.shape
        head_dim = dim // self.n_heads

        if self.single_matrix:
            q, k, v = self.QKV(x).chunk(3, dim=-1)
        else:
            k, q, v = self.K(x), self.Q(x), self.V(x)  # (bs, seq, hid)

        # split heads - process them independently, just Like different elements in the batch
        # (bs, seq, hid) -> (seq, bs * head, hid / head) -> (bs * head, seq, hid / head)
        k = k.transpose(0, 1).contiguous().view(seq, bsz * self.n_heads, head_dim).transpose(0, 1)
        q = q.transpose(0, 1).contiguous().view(seq, bsz * self.n_heads, head_dim).transpose(0, 1)
        v = v.transpose(0, 1).contiguous().view(seq, bsz * self.n_heads, head_dim).transpose(0, 1)

        alpha = F.softmax(k @ q.transpose(1, 2) / self.scale, dim=-1)  # (bs * head, seq, hid/head) @ (bs / head, hid / head, seq)

        attn = alpha @ v  # (bs * head, seq, seq) @ (bs * head, seq, hid / head)

        # (bs * head, seg, hid / head) -> (seq, bs * head, hid / head) ->  (seq, bs, hid) ->  (bs, seq, hid)
        attn = attn.transpose(0, 1).contiguous().view(seq, bsz, dim).transpose(0, 1)
        if self.has_out_proj:
            attn = self.out_proj(attn)
        return attn


class TransformerLayer(nn.Module):
    def __init__(self, dim=512, heads=2, girth=1, dropout=0.1,
                 single_matrix=True,
                 has_out_proj=True):
        super().__init__()
        self.attention = MultiHeadAttention(dim, n_heads=heads,
                                            single_matrix=single_matrix,
                                            has_out_proj=has_out_proj)

        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(dim, dim * girth)
        self.linear2 = nn.Linear(dim * girth, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # (batch * dims * sequence) => (batch * sequence * dims)
        x = x.permute(0, 2, 1).contiguous()

        attn = self.attention(x)
        x = x + self.dropout1(attn)
        x = self.norm1(x)

        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)

        # (batch * sequence * dims) => (batch * dims * sequence)
        x = x.permute(0, 2, 1).contiguous()
        return x


def load_pre_trained_transformer(original_decoder, new_decoder):
    """Load original PyTorch TransformerEncoder weights
    into a simplified transformer layer for transfer learning
    """
    assert len(original_decoder) == len(new_decoder)

    # assume only equal dims everywhere
    dim = new_decoder[0].linear1.in_features

    for i in range(0, len(original_decoder)):
        # load attention
        if hasattr(new_decoder[i].attention, 'out_proj'):
            print(f'Loading out proj layer {i}')
            new_decoder[i].attention.out_proj.load_state_dict(original_decoder[i].self_attn.out_proj.state_dict())

        if hasattr(new_decoder[i].attention, 'QKV'):  # fused matrix
            print(f'Loading fused matrix {i}')
            new_decoder[i].attention.QKV.weight = nn.Parameter(
                original_decoder[i].self_attn.state_dict()['in_proj_weight'].clone().detach()
            )
            new_decoder[i].attention.QKV.bias = nn.Parameter(
                original_decoder[i].self_attn.state_dict()['in_proj_bias'].clone().detach()
            )
        else:  # separate matrices
            # keep Q K V as separate matrices
            # looks like order is correct
            # https://github.com/zhangguanheng66/pytorch/blob/6c743c7721251ca9b5046fc56a071bc1f36916be/torch/nn/functional.py#L3182
            new_decoder[i].attention.Q.weight = nn.Parameter(
                original_decoder[i].self_attn.state_dict()['in_proj_weight'][:dim, :].clone().detach()
            )
            new_decoder[i].attention.Q.bias = nn.Parameter(
                original_decoder[i].self_attn.state_dict()['in_proj_bias'][:dim].clone().detach()
            )
            new_decoder[i].attention.K.weight = nn.Parameter(
                original_decoder[i].self_attn.state_dict()['in_proj_weight'][dim:dim * 2, :].clone().detach()
            )
            new_decoder[i].attention.K.bias = nn.Parameter(
                original_decoder[i].self_attn.state_dict()['in_proj_bias'][dim:dim * 2].clone().detach()
            )
            new_decoder[i].attention.V.weight = nn.Parameter(
                original_decoder[i].self_attn.state_dict()['in_proj_weight'][dim * 2:, :].clone().detach()
            )
            new_decoder[i].attention.V.bias = nn.Parameter(
                original_decoder[i].self_attn.state_dict()['in_proj_bias'][dim * 2:].clone().detach()
            )

        # load projection layers
        print(f'Loading linear1 linear2 {i}')
        new_decoder[i].linear1.load_state_dict(original_decoder[i].linear1.state_dict())
        new_decoder[i].linear2.load_state_dict(original_decoder[i].linear2.state_dict())

    return new_decoder