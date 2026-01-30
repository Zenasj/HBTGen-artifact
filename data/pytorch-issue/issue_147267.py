import torch.nn as nn

python
import argparse
import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

create_block_mask_compiled = torch.compile(create_block_mask)
flex_attention_compiled    = torch.compile(flex_attention)


def causal_mask_mod(b_idx, h_idx, q_idx, kv_idx):
    return q_idx >= kv_idx


def attn(feat, H_BlockMask):
    B,H,N,C = feat.shape
    block_mask = create_block_mask_compiled(mask_mod   = causal_mask_mod,
                                            B          = B,
                                            H          = H_BlockMask,
                                            Q_LEN      = N,
                                            KV_LEN     = N,
                                            device     = 'cuda',
                                            BLOCK_SIZE = 128)
    feat = flex_attention_compiled(feat, feat, feat, block_mask=block_mask)
    return feat


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--setting',
                        help = 'switches between experimental settings 0,...,2',
                        type = int)
    args = parser.parse_args()

    B =  64
    H =   8
    C =  32

    # For N>=128, flex_attention works as expected.
    if args.setting == 0:
        N = 129
        H_BlockMask = H
    # For N<128, we get RuntimeError: Triton Error [CUDA]: device-side assert triggered
    elif args.setting == 1:
        N = 127
        H_BlockMask = H
    # When setting H=1 in create_block_mask, everything works despite N<128.
    elif args.setting == 2:
        N = 127
        H_BlockMask = 1
    else:
        raise ValueError('Invalid setting passed, should be in 0,...,2.')

    feat = torch.randn(B,H,N,C).cuda()
    feat = attn(feat, H_BlockMask)
    print('attn:', feat.shape, feat[0,0,0,0])