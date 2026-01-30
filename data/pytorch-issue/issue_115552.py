import torch.nn as nn

import torch
import transformers


config = transformers.OpenLlamaConfig(
    vocab_size=8096, hidden_size=256, num_hidden_layers=2, num_attention_heads=2
)
batch, seq = 4, 256

def create_model() -> torch.nn.Module:
    return transformers.OpenLlamaModel(config).eval()

def create_args():
    return tuple()

def create_kwargs():
    input_ids = torch.randint(0, config.vocab_size, (batch, seq))
    attention_mask = torch.ones(batch, seq, dtype=torch.bool)
    position_ids = torch.arange(0, seq, dtype=torch.long)
    position_ids = position_ids.unsqueeze(0).view(-1, seq)
    return {"input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids}

model = create_model()
args = create_args()
kwargs = create_kwargs()

ep = torch.export.export(model, args=args, kwargs=kwargs)