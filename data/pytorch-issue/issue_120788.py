import torch.nn as nn

import torch
import torch.nn.functional as F
import math

def naive_sdp2(Q, K, V, mask=None):
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
    attn_probs = torch.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_probs, V)
    return output

query = torch.rand(1, 32, 1024, 64, dtype=torch.bfloat16)
key = torch.rand(1, 32, 1024, 64, dtype=torch.bfloat16)
value = torch.rand(1, 32, 1024, 64, dtype=torch.bfloat16)

Q = query.view(query.size(0) * query.size(1), query.size(2), query.size(3))

K = key.view(key.size(0) * key.size(1), key.size(2), key.size(3))

V = value.view(value.size(0) * value.size(1), value.size(2), value.size(3))

import time
repeat_time = 100
total_time = 0
for i in range(repeat_time):
    start_time = time.time()
    naive_sdp2(Q, K, V)
    total_time += time.time() - start_time

print(f"attn total time:{total_time}s, avg time: {total_time / repeat_time}s")

total_time = 0
for i in range(repeat_time):
   start_time = time.time()
   F.scaled_dot_product_attention(query,key,value)
   total_time += time.time() - start_time
print(f"fa total time:{total_time}s, avg time: {total_time / repeat_time}s")

from torch.profiler import profile, record_function, ProfilerActivity
num_iter = 5
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    for i in range(num_iter):
        naive_sdp2(Q, K, V)
print(prof.key_averages().table(sort_by="cpu_time_total"))