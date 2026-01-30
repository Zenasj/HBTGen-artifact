import torch.nn as nn

model = llama()
apply_prompt_tuning(model) # Adds an extra nn.Embedding parameter, for example
FSDP(model) # note: cannot call FSDP before prompt_tuning, as the embedding won't get sharded
model.load_state_dict(llama_checkpoint, strict=False) # pretrained model checkpoint