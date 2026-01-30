import torch.nn as nn

import os
import torch
import torch.distributed as dist
from timm.models.swin_transformer_v2 import SwinTransformerV2, WindowAttention

def main():
    # Initialize the distributed environment
    dist.init_process_group(backend='nccl')  # Use NCCL backend
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Set the device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    print(f"Running on rank {rank} out of {world_size} processes, device: {device}")

    # Compile WindowAttention.forward
    WindowAttention.forward = torch.compile(WindowAttention.forward)

    # Create the model
    model = SwinTransformerV2(
        img_size=(128, 128),
        depths=(2, 2, 18, 2),
        window_size=8,
        patch_size=4,
        embed_dim=128,
        num_heads=(4, 8, 16, 32),
        num_classes=0,
        in_chans=3
    ).to(device)

    # Wrap the model with DistributedDataParallel
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank] if torch.cuda.is_available() else None)

    # Create a sample input tensor
    im = torch.empty(1, 3, 128, 128).to(device)

    # Forward pass
    output = ddp_model(im)
    print(f"Rank {rank}: Output shape: {output.shape}")

    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    main()