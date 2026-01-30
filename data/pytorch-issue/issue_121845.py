import torch.nn as nn

score_mod(score[0,2,8,9], torch.tensor(0), torch.tensor(2), torch.tensor(8), torch.tensor(9))

import torch
from torch.nn.attention._templated_attention import _templated_attention as templated_attention

torch.manual_seed(0)

# Lets create some input tensors
# The input tensor has shape (batch_size, num_heads, seq_len, head_dim)
query = torch.randn(8, 8, 2048, 64, device="cuda", dtype=torch.float32)
key = torch.randn(8, 8, 2048, 64, device="cuda", dtype=torch.float32)
value = torch.randn(8, 8, 2048, 64, device="cuda", dtype=torch.float32)

# Lets create a fun new score_modification! I will call this
# Checkerboard. It will reduce the score for neighboring tokens (1 step apart)
# in the sequence. And increase the score for tokens 2 steps apart. For everything
# else, the score will remain the same.

def checkerboard(score, batch, head, token_q, token_kv):
    score = torch.where(torch.abs(token_kv - token_q) == 1, score * 0.5, score)
    score = torch.where(torch.abs(token_kv - token_q) == 2, score * 2.0, score)
    return score

# Lets call templated_attention with this new score modification
output = templated_attention(query, key, value, score_mod=checkerboard)

compiled_templated_attention = torch.compile(templated_attention)
out_compiled = compiled_templated_attention(query, key, value, score_mod=checkerboard)

torch.testing.assert_close(output, out_compiled, atol=2e-2, rtol=2e-2)