import torch
import torch.nn as nn
import traceback
import os, pickle
import numpy as np

import requests
import pickle

def load_pickle_from_url(url) :
  # Download the file
  response = requests.get(url)
  if response.status_code == 200:
      # Load the content of the file
      data = pickle.loads(response.content)

      return data
  else:
      print("Failed to download the file")
      return None

pickle_url = "https://github.com/GwiHwan-Go/repo/raw/main/issues/pickles/sub_diff.pkl"
inputs = load_pickle_from_url(pickle_url)

if inputs is None:
    raise ValueError("Pickle file could not be downloaded. Please check the URL.")

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, out, other, input):

        
        res = torch.sub(out=out, alpha=4, input=input, other=other)        
        return res


x = torch.rand([10,9,8],dtype=torch.float16)
input = inputs[0]['input'] #torch.rand([10,9,8],dtype=torch.float16)
y = torch.randint(-2147483648,2147483647,[10,9,8],dtype=torch.int32)
z = x.clone()# if use x instead of z, no bug.

model = Model().to(torch.device('cpu'))
eag = model(x, y, input)
opt = torch.compile(model.forward, mode='max-autotune')(z, y, input)

np.testing.assert_allclose(eag.to('cpu').numpy(), opt.to('cpu').numpy(), rtol=1e-3, atol=1e-3, equal_nan=True, verbose=True)

import torch
print(torch.__version__) # 2.3.0a0+git6b6fe48

def promote(t):
  t1 = t.to(dtype=torch.float16)
  return torch.isposinf(t1), torch.isneginf(t1)

t = torch.tensor([100000, -100000, 65503], dtype=torch.int32)

print(promote(t))  # (tensor([ True, False, False], device='cuda:0'), tensor([False,  True, False], device='cuda:0'))
print(torch.compile(promote, mode='max-autotune')(t))  # (tensor([False, False, False], device='cuda:0'), tensor([False, False, False], device='cuda:0'))

import torch

def sub(x, y):
   return torch.sub(x, y)

a = torch.tensor([1, -torch.inf, 3], dtype=torch.float16, device='cuda')
b = torch.tensor([100000, -100000, 65503], dtype=torch.int32, device='cuda')

# Discrepancy between eager and compile on the first index
print(sub(a, b))  # tensor([   -inf,     nan, -65504.], device='cuda:0', dtype=torch.float16)
print(torch.compile(sub)(a, b))  # tensor([   -inf,    -inf, -65504.], device='cuda:0', dtype=torch.float16)