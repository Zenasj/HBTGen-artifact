import torch

torch.load("path/to/model", map_location='cpu')
print(model.state_dict())

for mod in model.modules():
     if not hasattr(mod, "_non_persistent_buffers_set"):
            mod._non_persistent_buffers_set = set()