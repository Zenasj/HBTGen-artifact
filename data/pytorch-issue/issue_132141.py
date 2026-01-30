import torch
import torch.nn as nn
from torch.export import export, Dim

# Define output and target sizes
output_size = (5,)
target_size = (5,)

# Create the loss module
loss_module = nn.MSELoss()

# Create a dynamic batch size
batch = Dim("batch", min=2, max=1024)

# Specify that the first dimension of each input/target is that batch size
dynamic_shapes = ({0: batch}, {0: batch})

  # Export the loss module
exported = export(
      loss_module,
      args=(torch.randn(*(128, *output_size)), torch.randn(*(128, *target_size))),
      strict=True,
      dynamic_shapes=dynamic_shapes,
  )