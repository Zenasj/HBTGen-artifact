import torch

from transformers import AutoConfig, AutoModelForSeq2SeqLM

config = AutoConfig.from_pretrained("t5-large")
model = AutoModelForSeq2SeqLM(config=config)
model.load_state_dict(torch.load("checkpoint.pt"))