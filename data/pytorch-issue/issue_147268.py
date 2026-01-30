import torch.nn as nn

python
import argparse
import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

flex_attention_compiled = torch.compile(flex_attention)

def pool(feat, kernel_size, stride, padding, CONTIGUOUS_1, CONTIGUOUS_2, CLONE):
    B,H,N,C = feat.shape
    X = int(N**.5)
    feat = feat.reshape(B*H,X,X,C)   # reshape to square pixel grid, treat heads as batch dimension
    feat = feat.moveaxis(-1,1)       # (BH,C,X,X), as required for pytorch grid ops like pool2d
    if CONTIGUOUS_1: # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        feat = feat.contiguous()     # <<< REQUIRED TO PREVENT ILLEGAL MEMORY ACCESS ERROR !!! <<<<<
    feat = torch.nn.functional.max_pool2d(feat, kernel_size, stride, padding)
    feat = feat.moveaxis(1,-1)       # (BH,C,X',X')
    if CONTIGUOUS_2: # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        feat = feat.contiguous()     # <<< REQUIRED TO PREVENT ILLEGAL MEMORY ACCESS ERROR !!! <<<<<
    feat = feat.reshape(B,H,-1,C)    # (B,H,N',C), as required by flex_attention
    if CLONE: # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        feat = feat.clone()          # <<< REINTRODUCES ERROR EVEN WHEN .contiguous IS USED !!! <<<<
    return feat


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--setting',
                        help = 'switches between experimental settings 0,...,3',
                        type = int)
    args = parser.parse_args()

    # Everything works out when using .contiguous after both .moveaxis in pool
    if args.setting == 0:
        CONTIGUOUS_1 = True
        CONTIGUOUS_2 = True
        CLONE        = False
    # Switching off .contiguous yields:
    # RuntimeError: CUDA error: an illegal memory access was encountered
    # (this works when not compiling flex_attention)
    elif args.setting == 1:
        CONTIGUOUS_1 = False
        CONTIGUOUS_2 = False
        CLONE        = False
    # Adding a .clone() before passing features into flex_attention again leads to an illegal memory
    # error despite using .contiguous.
    elif args.setting == 2:
        CONTIGUOUS_1 = True
        CONTIGUOUS_2 = True
        CLONE        = True
    # When using only the first .contiguous, we get yet another error:
    # torch._inductor.exc.InductorError: LoweringException: AssertionError: Query must be contiguous in the last dimension
    elif args.setting == 3:
        CONTIGUOUS_1 = True
        CONTIGUOUS_2 = False
        CLONE        = True
    else:
        raise ValueError('Invalid setting passed, should be in 0,...,3.')


    B = 64
    H =  8
    C = 64
    N = 32**2

    feat = torch.randn(B,H,N,C).cuda()
    feat = pool(feat, kernel_size=5, stride=1, padding=0,
                CONTIGUOUS_1=CONTIGUOUS_1, CONTIGUOUS_2=CONTIGUOUS_2, CLONE=CLONE)
    print('pool:', feat.shape, feat[0,0,0,0])
    feat = flex_attention_compiled(feat, feat, feat)
    print('attn:', feat.shape, feat[0,0,0,0]) # accessing feat is required to surface the error