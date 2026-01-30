import torch
import time
from transformers import DistilBertModel, DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

model = DistilBertModel.from_pretrained("distilbert-base-uncased")

text = "This text is to test profile of huggingface's transfomers, especially distilbert model."

input_ids = torch.tensor([tokenizer.encode(text).ids])

with torch.autograd.profiler.profile() as prof:
    with torch.no_grad():
        for _ in range(30):
            output = model(input_ids)

print(prof.key_averages().table(sort_by="self_cpu_time_total"))

import torch
import time
from transformers import DistilBertModel, BertTokenizer, DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

model = DistilBertModel.from_pretrained("distilbert-base-uncased")

text = "This text is to test profile of huggingface's transfomers, especially distilbert model."

#input_ids = torch.tensor([tokenizer.encode(text).ids])
input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)

with torch.autograd.profiler.profile() as prof:
    with torch.no_grad():
        for _ in range(30):
            output = model(input_ids)

print(prof.key_averages().table(sort_by="self_cpu_time_total"))