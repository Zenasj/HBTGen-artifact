import torch.nn as nn

from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

class GPTBigCodeAttention(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.c_attn = nn.Linear(config.hidden_size, config.hidden_size + 2 * 15)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        print("attention_mask.huhu", attention_mask.huhu)
        print("attention_mask._padding_mask", attention_mask._padding_mask)
        query = self.c_attn(hidden_states)

        return query

class GPTBigCodeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.h = nn.ModuleList([GPTBigCodeAttention(config, layer_idx=i) for i in range(config.num_hidden_layers)])

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple:
        attention_mask = torch.Tensor([4, 5, 6])
        attention_mask._padding_mask = None

        attention_mask.huhu = "is this working?"

        hidden_states = self.wte(input_ids)
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                # None for past_key_value
                return module(*inputs)

            return custom_forward

        outputs = torch.utils.checkpoint.checkpoint(
            create_custom_forward(self.h[0]),
            hidden_states,
            attention_mask,
        )
        #outputs = self.h[0](hidden_states, attention_mask)

        return outputs


from transformers import AutoConfig, AutoTokenizer

cfg = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-GPTBigCodeForCausalLM")

model = GPTBigCodeModel(cfg)
tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-GPTBigCodeForCausalLM")

torch_device = "cuda"

inp = tokenizer("this is me", return_tensors="pt").to(torch_device)

model.to(torch_device)
model = model.train()

result = model(inp["input_ids"])

loss = result[0].sum()

print("call backward ------")
loss.backward()

class QTensor(torch.Tensor):
    @staticmethod 
    def __new__(cls, x, _padding_mask, *args, **kwargs): 
        return super().__new__(cls, x, *args, **kwargs) 
    
    def __init__(self, data, _padding_mask=False):
        self._padding_mask = _padding_mask