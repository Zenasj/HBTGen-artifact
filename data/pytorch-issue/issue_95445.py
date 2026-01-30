3
# git clone the repo or just download https://raw.githubusercontent.com/karpathy/nanoGPT/master/model.py

import torch
from model import GPT, GPTConfig

gpt = GPT(GPTConfig()).cuda()
gpt_compiled = torch.compile(gpt)
x = torch.ones((1, 1024), dtype=torch.int64).cuda()  # a fake input
gpt_compiled(x)  # throw errors below