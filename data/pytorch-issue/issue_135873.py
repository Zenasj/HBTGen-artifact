import torch

fn = torch.bucketize
compiled_fn = torch.compile(fn)

boundaries = torch.linspace(-1.0, 1.0, 33, device="cuda")
vals = torch.randn(16, device="cuda")
success = fn(vals, boundaries) == compiled_fn(vals, boundaries)
assert torch.all(success), "Contiguous check failed!"  # passes

noncontiguous_boundaries = boundaries[::2]
# This next line raises a warning about this exact issue from eager-mode bucketize.
success = fn(vals, noncontiguous_boundaries) == compiled_fn(vals, noncontiguous_boundaries)
assert torch.all(success), "Non-contiguous check failed!"  # fails

# This repro would also work with torch.searchsorted (once the lowering in
# https://github.com/pytorch/pytorch/pull/135701 merges, if the checks for continuity in
# the lowering are removed).