import torch
from transformers import LlamaForCausalLM, LlamaConfig

config = LlamaConfig(num_hidden_layers=2, 
                     attn_implementation='flash_attention_2')
model = LlamaForCausalLM(config=config)
model.half()
model.cuda()
optim_model = torch.compile(model, dynamic=True, backend='cudagraphs')

def test_ktimes(model, times, bs=4, mx=512):
    x = torch.randint(0, 32000, (bs, mx), device='cuda')
    for _ in range(times):
        y = model(x, labels=x)
    return None

# trigger first type
with torch.no_grad():
    optim_model.eval()
    print(test_ktimes(optim_model, 1, bs=16, mx=512))

# trigger second type
x = torch.randint(0, 32000, (16, 512), device='cuda')
y = optim_model(x, labels=x)

import torch
from transformers import LlamaForCausalLM, LlamaConfig

config = LlamaConfig(num_hidden_layers=2, 
                     attn_implementation='flash_attention_2')
model = LlamaForCausalLM(config=config)
model.half()
model.cuda()
optim_model = torch.compile(model, dynamic=True, mode='reduce-overhead')

def test_ktimes(model, times, bs=4, mx=512):
    x = torch.randint(0, 32000, (bs, mx), device='cuda')
    for _ in range(times):
        y = model(x, labels=x)
    return None

# trigger first type
with torch.no_grad():
    optim_model.eval()
    print(test_ktimes(optim_model, 1, bs=16, mx=512))

# trigger second type
x = torch.randint(0, 32000, (16, 512), device='cuda')
y = optim_model(x, labels=x)