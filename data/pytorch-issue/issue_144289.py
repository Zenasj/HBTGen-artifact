"""
torchrun --standalone --nproc_per_node=2 test.py
"""
import os
import torch
import torch.distributed as dist
from torch.distributed._tensor.api import DTensor
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
    TransformerBlock,
)
def main():
    torch.manual_seed(42)
    model_args = ModelArgs(
        n_layers=12,
        vocab_size=50304,
        n_heads=32,
        dim=2048,
        max_seq_len=2048,
        dropout_p=0.0,
    )
    model = Transformer(model_args)
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
    fsdp_cfg = {"mp_policy": mp_policy}
    for module in model.modules():
        if isinstance(module, TransformerBlock):
            fully_shard(module, **fsdp_cfg)
    fully_shard(model, **fsdp_cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)
    inp = torch.randint(0, model_args.vocab_size, (8, 1024), device="cuda")
    loss = model(inp).sum()

    for name, param in model.named_parameters():
        if not isinstance(param, DTensor):
            if dist.get_rank() == 0:
                print(f"Before backward Non-DTensor: {name}")

    loss.backward()

    for name, param in model.named_parameters():
        if not isinstance(param, DTensor):
            if dist.get_rank() == 0:
                print(f"After backward Non-DTensor: {name}")

    optim.step()

if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    gpu_id = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    rank = gpu_id
    main()
    dist.destroy_process_group()