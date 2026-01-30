import torch.nn as nn

import os
import time
import math

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.nn.attention.flex_attention import flex_attention

class Model(torch.nn.Module):
    def __init__(self, S, H, D):
        super().__init__()

        self.S = S
        self.H = H
        self.D = D

        alibi_bias = self.generate_alibi_bias(H)
        self.register_buffer("alibi_bias", alibi_bias, persistent=True)
        self.attention = flex_attention

        self.project_qk = torch.nn.Linear(H * D, H * D * 2)
        self.project_v = torch.nn.Linear(H * D, H * D)

    def forward(self, hidden_states):
        batch_size, _, _ = hidden_states.size()

        query, key = self.project_qk(hidden_states).chunk(2, dim=2)
        query = query.view(self.S, batch_size, self.H, self.D)
        query = query.permute(1, 2, 0, 3)

        key = key.view(self.S, batch_size, self.H, self.D)
        key = key.permute(1, 2, 0, 3)

        value = self.project_v(hidden_states)
        value = value.view(self.S, batch_size, self.H, self.D)
        value = value.permute(1, 2, 0, 3)

        return self.attention(query, key, value, score_mod=self.alibi_score_mod)

    def generate_alibi_bias(self, num_heads):
        alibi_bias = [math.exp2(-((i + 1) * 8.0) / num_heads) for i in range(num_heads)]
        return torch.tensor(alibi_bias)

    def alibi_score_mod(self, score, b, h, q_idx, kv_idx):
        bias = (q_idx - kv_idx) * self.alibi_bias[h]
        return score + bias

if __name__ == "__main__":

    B = 64
    H = 12
    S = 512
    D = 64

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    model = Model(S, H, D)
    model.to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank])
    torch.compile(model)

    for i in range(100):
        start = time.perf_counter()
        hidden_states = torch.randn(B, S, H * D).to(device)
        attention_scores = model(hidden_states)
        torch.cuda.synchronize()
        print(f"{i}: {time.perf_counter() - start:.4f}")