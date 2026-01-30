import torchvision

import torch
from transformers import BertForSequenceClassification, BertConfig

config = BertConfig()
model = BertForSequenceClassification(config=config).eval().cuda()

for k, v in model.named_parameters():
    v.data = 0.1 * torch.randn_like(v.data)
    
inputs_embeds = torch.randn([1, 100, config.hidden_size], requires_grad=True).cuda()
attention_mask = torch.ones([1, 100]).cuda()

outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)[0]
grads1 = torch.autograd.grad(outputs[0, 1], inputs_embeds, create_graph=False)[0]

outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)[0]
grads2 = torch.autograd.grad(outputs[0, 1], inputs_embeds, create_graph=True)[0]

print((grads1 - grads2).abs().max())
assert not ((grads1 - grads2).abs() > 1e-6).any()

import torch
from torchvision.models import resnet50

model = resnet50().eval().cuda()
x = torch.randn([1, 3, 224, 224],requires_grad=True).cuda()

for k, v in model.named_parameters():
    v.data = 0.1 * torch.randn_like(v.data)
    
outputs = model(x)
grads1 = torch.autograd.grad(outputs[0, 1], x, create_graph=False)[0]

outputs = model(x)
grads2 = torch.autograd.grad(outputs[0, 1], x, create_graph=True)[0]

print((grads1 - grads2).abs().max())
assert not ((grads1 - grads2).abs() > 1e-6).any()