import torch

swaps = copy(tensors)
for i, t in enumerate(tensors):
    swaps[i] = tensor.view(torch.uint8).view(-1)
torch.stack(swaps, storage)
out = storage.split(tensor_lengths)
out = [t.view(t_src.dtype).view(t_src.shape) for t, t_src in zip(out, tensors)]

torch._dynamo.maybe_mark_dynamic(values, 0)