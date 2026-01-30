import torch.nn as nn

from torch.nn.attention.flex_attention import flex_attention, create_block_mask, _DEFAULT_SPARSE_BLOCK_SIZE
import torch
import argparse
import math
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--seq_len", type=int, default=32*1024)
parser.add_argument("--head_num", type=int, default=32)
parser.add_argument("--head_dim", type=int, default=128)
parser.add_argument("--chunk_size", type=int, default=2*1024)
args = parser.parse_args()

flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune")

def get_dynamic_mod(recent_token_num):
    def get_mask(b, h, q_idx, kv_idx):
        recent_mask = kv_idx < recent_token_num
        real_kv_idx = kv_idx - recent_token_num
        casual_mask = q_idx >= real_kv_idx

        return recent_mask | casual_mask

    return get_mask

@torch.no_grad
def main():
    q = torch.randn(1, args.head_num, args.seq_len, args.head_dim, dtype=torch.bfloat16).cuda()
    k = torch.randn(1, args.head_num, args.seq_len, args.head_dim, dtype=torch.bfloat16).cuda()
    v = torch.randn(1, args.head_num, args.seq_len, args.head_dim, dtype=torch.bfloat16).cuda()
    iter_num = math.ceil(args.seq_len / args.chunk_size)
    num_past_tokens = 0
    for i in tqdm(range(iter_num)):
        query_states = q[:, :, i*args.chunk_size:(i+1)*args.chunk_size, :]
        key_states = k[:, :, i*args.chunk_size-num_past_tokens:(i+1)*args.chunk_size, :]
        value_states = v[:, :, i*args.chunk_size-num_past_tokens:(i+1)*args.chunk_size, :]

        print(query_states.shape, key_states.shape, value_states.shape)

        mask_mod = get_dynamic_mod(num_past_tokens)

        # wheter to use `_compile=True` here is important!
        block_mask = create_block_mask(mask_mod, 1, 1, args.chunk_size, args.chunk_size+num_past_tokens, device="cuda", BLOCK_SIZE=(128, 64), _compile=True)

        attn_output = flex_attention(query_states, key_states, value_states, block_mask=block_mask)

        num_past_tokens = args.chunk_size * (i+1)
        # num_past_tokens = 0


if __name__ == "__main__":
    main()