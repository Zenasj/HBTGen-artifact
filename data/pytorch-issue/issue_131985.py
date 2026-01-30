import torch.nn as nn

import torch
from torch import nn

with torch.inference_mode():
    # Set seed for reproducibility
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Parameters
    in_features = 64
    height, width = 128, 128
    stride = 1

    # Loop through different batch sizes
    for N in range(1, 48):
        batch_size = 8 * N
        
        # Create identical input batch
        x = torch.randn(batch_size, in_features, height, width).cuda()
        x[:] = x[0]

        # Define and apply convolutional layer
        conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3, padding=1, stride=stride).cuda()
        y = conv2(x)

        # Check if all examples in the output batch are identical
        input_identical = (x[:1,...] == x).all()
        output_identical = (y[:1,...] == y).all()

        # Identify mismatched examples and the first position of mismatch
        mismatched_examples = (y[:1,...] != y).any(dim=1).any(dim=1).any(dim=1)
        num_mismatched = mismatched_examples.sum().item()
        first_mismatch_position = torch.nonzero(mismatched_examples).flatten().tolist()

        # Print results for each batch size in a single line
        print(f"Batch Size: {batch_size:03d} | Input Identical: {str(input_identical.item()):>5} | Output Identical: {str(output_identical.item()):>5} | Num Mismatched: {num_mismatched} | First Mismatch Position: {first_mismatch_position[0] if num_mismatched > 0 else 'N/A'} | Shape: {x.shape}")