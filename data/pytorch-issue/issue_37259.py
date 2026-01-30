import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=1)

def train_async(data):
    with torch.no_grad():
        torch.zeros(64, 8, 2, 128)
            
while True:
    executor.submit(train_async, []).result()

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=1)

def train_async(data):
    with torch.no_grad():
        torch.zeros(64, 8, 2, 128)
            
while True:
    train_async([])