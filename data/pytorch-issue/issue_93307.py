import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "google/flan-t5-base"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = torch.compile(model)

from torch import _dynamo
_dynamo.config.verbose = True
_dynamo.explain(model)