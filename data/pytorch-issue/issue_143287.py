import torch
import torch.nn as nn

py
def generate_causal_mask(seq_length):
    # Diagonal = 0, so each element attends only to elements before it, excluding itself
    mask = torch.triu(torch.full((seq_length, seq_length), 1, dtype=torch.float32), diagonal=0).bool()
    
    # Allow the first element to attend to itself to avoid NaN results
    mask[0, 0] = False
    return mask

tensor([[False,  True,  True,  True,  True,  True,  True,  True],
        [False,  True,  True,  True,  True,  True,  True,  True],
        [False, False,  True,  True,  True,  True,  True,  True],
        [False, False, False,  True,  True,  True,  True,  True],
        [False, False, False, False,  True,  True,  True,  True],
        [False, False, False, False, False,  True,  True,  True],
        [False, False, False, False, False, False,  True,  True],
        [False, False, False, False, False, False, False,  True]])

py
if __name__ == "__main__":
    embed_dim = 16
    batch_size = 1
    seq_len = 8

    mha = nn.MultiheadAttention(embed_dim, num_heads=1, batch_first=True)

    x = torch.randn(batch_size, seq_len, embed_dim).requires_grad_(True)

    causal_mask = generate_causal_mask(seq_len)

    print(causal_mask)

    output, _ = mha(x, x, x, attn_mask=causal_mask)

    # Gradient of the output with respect to the token at position t
    t = 5
    loss = output[:, t].sum().backward()
    print("Gradient of the token:")
    print(x.grad)