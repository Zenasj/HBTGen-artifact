import torch.nn as nn

# bug_demo.py

import torch

n_trials = 100
for ii in range(n_trials):
    a = torch.randn(1024, device='mps')
    b = torch.randn(499, device='mps')
    c = torch.nn.functional.conv1d(a.view(1, 1, -1), b.view(1, 1, -1))
    if torch.isnan(torch.sum(c)):
        print(f'mps: trial {ii}, nan elements {torch.isnan(c.squeeze()).nonzero().view(-1).cpu().numpy()}')
        
for ii in range(n_trials):
    a = torch.randn(1024, device='cpu')
    b = torch.randn(499, device='cpu')
    c = torch.nn.functional.conv1d(a.view(1, 1, -1), b.view(1, 1, -1))
    if torch.isnan(torch.sum(c)):
        print(f'cpu: trial {ii}, elements {torch.isnan(c.squeeze()).nonzero().view(-1).numpy()}')

# bug_demo_conv2d.py
import torch

n_trials = 100
for ii in range(n_trials):
    a = torch.randn(1024, device='mps')
    b = torch.randn(499, device='mps')
    c = torch.nn.functional.conv2d(a.view(1, 1, 1, -1), b.view(1, 1, 1, -1))
    if torch.isnan(torch.sum(c)):
        print(f'mps: trial {ii}, nan elements {torch.isnan(c.cpu().squeeze()).nonzero().view(-1).numpy()}')

for ii in range(n_trials):
    a = torch.randn(1024, device='cpu')
    b = torch.randn(499, device='cpu')
    c = torch.nn.functional.conv2d(a.view(1, 1, 1, -1), b.view(1, 1, 1, -1))
    if torch.isnan(torch.sum(c)):
        print(f'cpu: trial {ii}, nan elements {torch.isnan(c.squeeze()).nonzero().view(-1).numpy()}')

# bug_demo_Conv2d.py
import torch

n_trials = 100
b_len = 499
conv_b = torch.nn.Conv2d(1, 1, (1, b_len), bias=False, device='mps')
for ii in range(n_trials):
    a = torch.randn(1, 1, 1, 1024, device='mps')
    conv_b.weight = torch.nn.Parameter(torch.randn(1, 1, 1, b_len, device='mps'), requires_grad=False)
    c = conv_b(a)
    if torch.isnan(torch.sum(c)):
        print(f'mps: trial {ii}, nan elements {torch.isnan(c.cpu().squeeze()).nonzero().view(-1).numpy()}')

conv_b = torch.nn.Conv2d(1, 1, (1, b_len), bias=False, device='cpu')
for ii in range(n_trials):
    a = torch.randn(1, 1, 1, 1024, device='cpu')
    conv_b.weight = torch.nn.Parameter(torch.randn(1, 1, 1, b_len, device='cpu'), requires_grad=False)
    c = conv_b(a)
    if torch.isnan(torch.sum(c)):
        print(f'cpu: trial {ii}, nan elements {torch.isnan(c.squeeze()).nonzero().view(-1).numpy()}')

for ii in range(100):
    fa = torch.randn(499, 526, device='cpu')
    b = torch.randn(1, 499, device='cpu')
    c = torch.mm(b, fa)
    if torch.isnan(torch.sum(c)):
        print(f'cpu: trial {ii}, elements {torch.isnan(c.squeeze()).nonzero().view(-1).numpy()}')
        fail_here()