import torch

py
with tempfile.NamedTemporaryFile() as f:
    with torch.TensorAutoRemoveContext():
         t = torch.from_file(f, shared=False, size=size, dtype=torch.uint8)
    # When we exited from the context, we explicitly delete the tensors created within the context
    # In other words, this is equivalent to `del t` here.