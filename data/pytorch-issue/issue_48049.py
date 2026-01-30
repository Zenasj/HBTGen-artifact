#!/usr/bin/env python

import os
import sys
import torch
os.environ["USE_TF"] = "0"
sys.path.insert(1, "src")

# !pip install ipyexperiments 
from ipyexperiments.utils.mem import gpu_mem_get_used_mbs, gpu_mem_get_used_no_cache_mbs

from transformers import BartForConditionalGeneration

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = BartForConditionalGeneration.from_pretrained('sshleifer/student_cnn_12_6').to(device)
model.eval()

vocab_size = 50264 # model.config.vocab_size
length = 10

AUTOCAST = False if "-f" in sys.argv else True
print(f"autocast: {AUTOCAST}")

class MemReport():
    def __init__(self, gc_collect=True):
        self.get_mem = gpu_mem_get_used_no_cache_mbs if gc_collect else gpu_mem_get_used_mbs
        self.cur = self.get_mem()
    def delta(self, id):
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        print(f"{id}: {gpu_mem_get_used_mbs()-self.cur}MB (peak {peak>>20}MB)")
        self.cur = self.get_mem()
        
mr = MemReport(gc_collect=False)

### reproducible code starts here ###

@torch.no_grad()
def logic():
    input_ids = torch.randint(vocab_size, (1,length)).to(device)
    mr.delta(0)
    for i in range(1,10):
        outputs = model(input_ids)
        mr.delta(i)

if AUTOCAST:
    with torch.cuda.amp.autocast():
        logic()
else:
    logic()

with torch.no_grad():
    print(a.requires_grad) # True
    b = a.clone()
    print(b.requires_grad) # False
    b = a.t()
    print(b.requires_grad) # True
    b = a.view(-1)
    print(b.requires_grad) # True

base = torch.rand(1, 5)
slice1 = base[0]
slice2 = base[0]

# This makes slice1 and base properly require gradients
slice1.copy_(torch.rand(5, requires_grad=True))
# Other views need to also be properly updated by reading the base's status
loss = slice2.sum()
loss.backward()