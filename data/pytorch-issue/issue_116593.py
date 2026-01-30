import torch

py
model = Model()
compiled_model = torch.compile(model)

model.load_state_dict(compiled_model.state_dict())  # previously key mismatch!

py
FSDP(torch.compile(model))
# or
DDP(torch.compile(model))