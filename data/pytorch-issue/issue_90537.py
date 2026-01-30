import torch
from transformers import BartTokenizer, BartForConditionalGeneration

device = torch.device('cuda')
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
model = model.to(device)
model = torch.compile(model)

inputs = tokenizer(
    "Summarize: You may want to stick it to your boss and leave your job, but don't do it if these are your reasons.",
    return_tensors="pt",
)
labels = tokenizer("Bad Reasons To Quit Your Job", return_tensors="pt")["input_ids"]

input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)
labels = labels.to(device)

loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss

import torch
from transformers import BartTokenizer, BartForConditionalGeneration

device = torch.device('cuda')
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
model = model.to(device)
model = torch.compile(model)

inputs = tokenizer(
    "Summarize: You may want to stick it to your boss and leave your job, but don't do it if these are your reasons.",
    return_tensors="pt",
)
labels = tokenizer("Bad Reasons To Quit Your Job", return_tensors="pt")["input_ids"]

input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)
labels = labels.to(device)

loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss

print(loss)