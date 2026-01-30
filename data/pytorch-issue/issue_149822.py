import torch
x = torch.ones(10)
# If CUDA is available, use a CUDA tensor for the output.
if torch.cuda.is_available():
    out_values = torch.empty(10, device="cuda")
    out_indices = torch.empty(10, dtype=torch.long, device="cpu")
    torch.max(x, 0, out=(out_values, out_indices))