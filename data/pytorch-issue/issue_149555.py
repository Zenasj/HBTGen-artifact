import torch

# Enable tensor returns for dynamo compatibility
torch.cuda.nvtx.enable_tensor_returns()

# Use NVTX functions in dynamo-compiled code
# All functions now return tensors

# with torch.compile context
with torch.cuda.nvtx.range("my_range"):
    pass

# Disable tensor returns to restore original behavior
torch.cuda.nvtx.disable_tensor_returns()