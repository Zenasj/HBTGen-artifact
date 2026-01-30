import torch.nn as nn
import torch.nn.functional as F

import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from torch.nn.attention.flex_attention import flex_attention
# flex_attention = torch.compile(flex_attention, dynamic=False)
 
torch.set_default_device("cuda")
torch.manual_seed(0)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.fc_q = nn.Linear(256, 256)
        self.fc_k = nn.Linear(256, 256)
        self.fc_v = nn.Linear(256, 256)
        self.fc_o = nn.Linear(256, 5)


    def forward(self, x):
        q = self.fc_q(x)
        k = self.fc_k(x)
        v = self.fc_v(x)

        q = q.view(1, 512, 16, 16).transpose(1, 2)
        k = k.view(1, 512, 16, 16).transpose(1, 2)
        v = v.view(1, 512, 16, 16).transpose(1, 2)

        out = self.fc_o(flex_attention(q, k, v).transpose(1, 2).reshape(1, 512, 256))
        return out


def main():
    x = torch.randn((1, 512, 256), requires_grad=True).cuda()
    y = torch.randint(0, 5, size=(5,)).cuda()

    model = Model().cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler()

    for epoch in tqdm(range(100)):
        optimizer.zero_grad()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            predictions = model(x).squeeze().sum(dim=0)
            loss = F.cross_entropy(predictions, y.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


if __name__ == "__main__":
    main()

import os
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'

import torch
from torch.nn.attention.flex_attention import flex_attention

if __name__ == "__main__":

    torch.set_default_device("cuda")
    torch.manual_seed(0)
    torch._dynamo.config.cache_size_limit = 1000

    B = 16
    H = 8
    S = 1024
    D = 64
    data_type = torch.float16 # <-------- float32 works
    device = "cuda"

    qkv = [
        torch.randn(B, H, S, D, device=device, dtype=data_type, requires_grad=True)
        for _ in range(3)
    ]

    flex_attention_c = torch.compile(flex_attention, dynamic=False)
    out = flex_attention_c(*qkv)