Python
import torch
import torch.nn as nn
import functools
from torch.nn.attention.flex_attention import flex_attention, create_block_mask


class AttentionModel(nn.Module):
    def __init__(self, initial_kv_len):
        super().__init__()
        self.kv_len = initial_kv_len
        self.q_len = 1

    def causal_mask_decode(self, b, h, q_idx, kv_idx):
        offset = self.kv_len - self.q_len
        return offset + q_idx >= kv_idx

    def forward(self, queries, keys, values, mask):
        self.kv_len = keys.shape[-2]
        bs, nh, seq_len, _ = queries.shape

        attention = functools.partial(flex_attention, block_mask=mask, enable_gqa=True)
        attention = torch.compile(attention)
        attn_output = attention(queries, keys, values)

        return attn_output


# Driver code
def main():
    # Set up parameters
    d_model = 256
    q_heads = 32
    kv_heads = 8
    kv_len = 128
    q_len = 1
    batch_size = 1

    # Initialize the model
    model = AttentionModel(kv_len)
    mask = create_block_mask(
        lambda a, b, c, d: model.causal_mask_decode(a, b, c, d), 1, 1, q_len, kv_len
    )

    # Create sample input tensors
    queries = torch.randn(batch_size, q_heads, q_len, d_model, device="cuda")
    keys = torch.randn(batch_size, kv_heads, kv_len, d_model, device="cuda")
    values = torch.randn(batch_size, kv_heads, kv_len, d_model, device="cuda")

    # Forward pass
    output = model(queries, keys, values, mask)

    print(f"Input shapes:")
    print(f"  Queries: {queries.shape}")
    print(f"  Keys: {keys.shape}")
    print(f"  Values: {values.shape}")
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    main()