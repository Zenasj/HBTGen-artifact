import numpy as np

def sample_temp(logits, temperature):
    probs = torch.exp(logits / temperature)
    return torch.multinomial(probs, 1)

import torch
x = torch.Tensor([0, float('inf')])
print(torch.multinomial(x, 1))
print(torch.multinomial(x.to('cuda'), 1))

def sample(logits, temperature=1.0):
    if temperature == 0:
        return torch.argmax(logits, dim=-1)
    noise = torch.FloatTensor(logits.shape).to(logits.device)
    noise.uniform_(1e-5, 1-1e-5)
    return torch.argmax(logits / temperature - torch.log(-torch.log(noise)), dim=-1)

def multinomial(probs=None, logits=None, temperature=None):
    if probs:
        logits = torch.log(probs)
    if temperature:
        logits /= temperature
    logits = torch.min(logits, torch.Tensor([40.]))
    probs = torch.exp(logits)
    return torch.multinomial(probs, 1)

def multinomial(probs=None, logits=None, temperature=1, num_samples=1,
                     min_prob=1e-20, max_logit=1e+20,
                     min_temperature=1e-20, max_temperature=1e+20):
    if probs is not None:
        probs = probs.clamp(min=min_prob)
        logits = torch.log(probs)
    logits = logits.clamp(max=max_logit)
    temperature = np.clip(temperature, min_temperature, max_temperature)
    logits = (logits - logits.max()) / temperature
    probs = torch.exp(logits)
    return torch.multinomial(probs, num_samples)

test_cases = [
    [0,0,0],
    [1,1,1],
    [1,0,float('+inf')],
    [1,0,float('-inf')],
    [1,0,1e+20],
    [1,0,1e-20],
    [1,0,-1e+20],
    [1,0,-1e-20]
]

for temperature in [float('-inf'), 0, 1e-20, 1, 1e+20, float('+inf')]: 
    for test_case in test_cases:
        for device in ['cpu', 'cuda']:
            x = torch.Tensor(test_case)
            print(multinomial(probs=x.to(device), temperature=temperature))
            print(multinomial(logits=x.to(device), temperature=temperature))