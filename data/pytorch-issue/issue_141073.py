import torch

batch_size = 32
chunks = 4
example_mb = torch.randn(batch_size // chunks, in_dim, device=device)
pipe = pipeline(mn, mb_args=(example_mb,), split_spec=split_spec)

stage = pipe.build_stage(rank, device)