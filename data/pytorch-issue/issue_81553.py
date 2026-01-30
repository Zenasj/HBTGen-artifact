# ran with "python test_ltc_only_torch.py --device=lazy --sync=1 --nvtx=1"
import torch

import torch._lazy
from torch._lazy.ts_backend import init as init_ts_backend
init_ts_backend()
torch.manual_seed(42)
from transformers import BertForSequenceClassification

def parse_args():
  import argparse
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--device', type=str, default='cuda')
  parser.add_argument('--sync', type=bool, default=False)
  parser.add_argument('--nvtx', type=bool, default=False)
  return parser.parse_args()

args = parse_args()

device = args.device
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True)

from transformers import AdamW
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text_batch = ["I love Pixar.", "I don't care for Pixar."]
encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
input_ids = encoding['input_ids'].to(device)
attention_mask = encoding['attention_mask'].to(device)

model = model.to(device)
model.train()

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
labels = torch.tensor([1,0]).unsqueeze(0).to(device)
for _ in range(6):
  torch.cuda.nvtx.range_push(f'Iter{_}')

  torch.cuda.nvtx.range_push('F')
  outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
  if args.sync:
    torch._lazy.mark_step()
    torch._lazy.wait_device_ops()
  torch.cuda.nvtx.range_pop()

  loss = outputs.loss

  torch.cuda.nvtx.range_push('B')
  optimizer.zero_grad()
  loss.backward()
  if args.sync:
    torch._lazy.mark_step()
    torch._lazy.wait_device_ops()
  torch.cuda.nvtx.range_pop()

  torch.cuda.nvtx.range_push('O')
  optimizer.step()
  if args.sync:
    torch._lazy.mark_step()
    torch._lazy.wait_device_ops()
  torch.cuda.nvtx.range_pop()

  torch.cuda.nvtx.range_pop()
torch._lazy.mark_step()
torch._lazy.wait_device_ops()