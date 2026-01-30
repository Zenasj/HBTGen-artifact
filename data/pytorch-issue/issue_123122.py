import torch

# Export
exported = torch.export.export(bert, (), kwargs=example_inputs)

# Unflatten
unflattened = torch.export.unflatten(exported)
out_unflat = unflattened(**example_inputs)