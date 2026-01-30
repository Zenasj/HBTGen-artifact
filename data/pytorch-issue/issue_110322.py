import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.q = nn.Linear(1024, 1024)
        self.k = nn.Linear(1024, 1024)
        self.v = nn.Linear(1024, 1024)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        queries = self.q(x).view(batch_size, seq_len, 8, 128).transpose(2, 1)
        keys = self.k(x).view(batch_size, seq_len, 8, 128).transpose(2, 1)
        values = self.v(x).view(batch_size, seq_len, 8, 128).transpose(2, 1)

        attn = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
        )

        return attn


model = Model().cuda().half()
model = torch.compile(model, dynamic=True)

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

input1 = torch.rand(5, 512, 1024, device="cuda", dtype=torch.float16)
input2 = torch.rand(5, 513, 1024, device="cuda", dtype=torch.float16)
input3 = torch.rand(5, 514, 1024, device="cuda", dtype=torch.float16)

out1 = model(input1)
out2 = model(input2)
out3 = model(input3)