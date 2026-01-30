import torch.nn as nn

outputs = model(**inputs)

for step, inputs in enumerate(epoch_iterator):
                    ...
                    profiler.step()

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

mname = "patrickvonplaten/t5-tiny-random"
model = T5ForConditionalGeneration.from_pretrained(mname).cuda()
tokenizer = T5Tokenizer.from_pretrained(mname)

input_ids = tokenizer('translate English to German: The house is wonderful.', return_tensors='pt').input_ids.cuda()
labels = tokenizer('Das Haus ist wunderbar.', return_tensors='pt').input_ids.cuda()
inputs = dict(input_ids=input_ids, labels=labels)

with torch.profiler.profile() as profiler:
    print("starting")
    outputs = model(**inputs)
    print("finished")
    #profiler.step()

import torch
from torch import nn
with torch.profiler.profile() as profiler:
    print("starting")
    embedding = nn.Embedding(10, 3).cuda()
    input = torch.LongTensor([[1,2,4,5],[4,3,2,9]]).cuda()
    embedding(input)
    print("finished")

# tutorial - non-CUDA
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

# my hanging example - CUDA
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

mname = "patrickvonplaten/t5-tiny-random"
model = T5ForConditionalGeneration.from_pretrained(mname).cuda()
tokenizer = T5Tokenizer.from_pretrained(mname)

input_ids = tokenizer('translate English to German: The house is wonderful.', return_tensors='pt').input_ids.cuda()
labels = tokenizer('Das Haus ist wunderbar.', return_tensors='pt').input_ids.cuda()
inputs = dict(input_ids=input_ids, labels=labels)

with torch.profiler.profile() as profiler:
    print("starting")
    outputs = model(**inputs)
    print("finished")
    #profiler.step()

#1  __pthread_rwlock_wrlock_full (abstime=0x0, clockid=0, rwlock=0x5567c675e9d0) at pthread_rwlock_common.c:830
#2  __GI___pthread_rwlock_wrlock (rwlock=0x5567c675e9d0) at pthread_rwlock_wrlock.c:27

import torch
with torch.profiler.profile() as profiler:
        pass

import torch
with torch.profiler.profile() as profiler:
        pass

import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

mname = "patrickvonplaten/t5-tiny-random"
model = T5ForConditionalGeneration.from_pretrained(mname).cuda()
tokenizer = T5Tokenizer.from_pretrained(mname)

input_ids = tokenizer('translate English to German: The house is wonderful.', return_tensors='pt').input_ids.cuda()
labels = tokenizer('Das Haus ist wunderbar.', return_tensors='pt').input_ids.cuda()
inputs = dict(input_ids=input_ids, labels=labels)

with torch.profiler.profile() as profiler:
    print(f"starting {os.getpid()}")
    outputs = model(**inputs)
    print("finished")

import torch
with torch.profiler.profile() as profiler:
        pass
# my normal profile code follows

import torch
x = torch.ones(1).cuda()
with torch.profiler.profile() as profiler:
        pass
# my normal profile code follows

with torch.profiler.profile() as profiler:
        pass

import torch
from torch import nn

# uncomment to remove the hanging
#with torch.profiler.profile() as profiler:
#    pass

scores = torch.rand(16,8,12,12).cuda()
with torch.profiler.profile() as prof:
    attn_weights = nn.functional.softmax(scores, dim=-1)

# uncomment to remove the hanging
#with torch.profiler.profile() as profiler:
#    pass

#uncomment to remove the hanging
#tmp = torch.empty(1, device="cuda")

import torch
from torch import nn

tmp = torch.empty(1, device="cuda")

scores = torch.rand(16,8,12,12).cuda()
with torch.profiler.profile() as prof:
    attn_weights = nn.functional.softmax(scores, dim=-1)