import torch

if attention_mask.shape[0] == 1 and not torch.any(attention_mask -1):
  position_ids = torch.arange(attention_mask.shape[-1], dtype=torch.long, 
                              device=attention_mask.device).expand(attention_mask.shape)
else:
  print(attention_mask)
  position_ids = attention_mask.cumsum(-1)

outputs = model.generate(input_ids=input_ids, do_sample=False, use_cache=False, max_new_tokens=max_length)