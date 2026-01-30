import torch
from model import GPT, GPTConfig

model = GPT(config=GPTConfig(n_layer=6, n_head=8, n_embd=1024, dropout=0,  bias=False)).cuda()

batch_size = 2
x = torch.randint(0, 50257, (batch_size, 1024)).cuda()
target = torch.randint(0, 50257, (batch_size, 1024)).cuda()

with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
    logits, loss = model(x, target)

loss.backward()

with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16) as autocast, torch.backends.cuda.sdp_kernel(enable_flash=False) as disable :
    logits, loss = model(x, target)