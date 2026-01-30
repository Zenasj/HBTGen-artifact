import torch.nn as nn

from transformers import BertForMaskedLM
import torch

batch_size = 7
device = torch.device("cuda:0")

model = BertForMaskedLM.from_pretrained("bert-base-uncased")
model.to(device)
model.train()

input_ids = torch.ones((batch_size, 512)).to(torch.int64).to(device)
attention_mask = torch.ones((batch_size, 512)).to(torch.int64).to(device)
labels = torch.ones((batch_size, 512)).to(torch.int64).to(device)
outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
loss = outputs.loss
loss.backward()
print(loss)

import torch

batch_size = 7
seq_length = 512
device = torch.device("cuda:0")

token_type_embeddings = torch.nn.Embedding(2, 768).to(device)

token_type_ids = torch.zeros(1, seq_length, dtype=torch.long, device=device).expand(batch_size, seq_length)
outputs = token_type_embeddings(token_type_ids)
loss = outputs.sum()
loss.backward()
print(loss)