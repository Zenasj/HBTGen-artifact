python
from config import GPT2_PATH # the path to GPT2 checkpoint
from transformers import GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from typing import Any
from functools import partial

def collate(tokeniser: PreTrainedTokenizer, max_length: int, batch: list) -> Any:
    batch = [sample['text'] for sample in batch]
    return tokeniser.batch_encode_plus(batch, padding=True, max_length=max_length, truncation=True, return_tensors='pt')

tokeniser: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(GPT2_PATH)
tokeniser.pad_token = tokeniser.eos_token_id
dataset = load_dataset('c4', 'en.noblocklist', split='train', streaming=True)
dataloader = DataLoader(dataset, batch_size=1024, num_workers=12, drop_last=True, collate_fn=partial(collate, tokeniser, 100), prefetch_factor=4)

i = 0
for batch in dataloader:
    i += 1
    if i > 100:
        break

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

dataset = datasets.FakeData(size=10000, transform=trans)


loader = torch.utils.data.DataLoader(
    dataset, batch_size=8, shuffle=True,
    num_workers=12, pin_memory=True, sampler=None)

i = 0
for d in loader:
    print("Batch {}".format(i))
    i += 1

transformers

torch

python
import transformers # imported but not used

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

dataset = datasets.FakeData(size=10000, transform=trans)


loader = torch.utils.data.DataLoader(
    dataset, batch_size=128, shuffle=True,
    num_workers=12, sampler=None)

i = 0
for d in loader:
    print("Batch {}".format(i))
    i += 1
# take 23.6 seconds

torch

transformers

python
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import transformers # imported after the torch

trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

dataset = datasets.FakeData(size=10000, transform=trans)


loader = torch.utils.data.DataLoader(
    dataset, batch_size=128, shuffle=True,
    num_workers=12, sampler=None)

i = 0
for d in loader:
    print("Batch {}".format(i))
    i += 1
# takes only 5.4 seconds

torch