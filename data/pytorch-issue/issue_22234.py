import torch.nn as nn

import torch

# Generate data.
logits = torch.normal(mean=torch.zeros((20, 10, 5), dtype=torch.float32))
logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
targets = torch.randint(1, 10, size=(10, 4), dtype=torch.int32)
target_lengths = torch.randint(1, 5, size=(10,), dtype=torch.int32)

# All examples have the same input length, so that cuDNN can be used.
input_lengths = 20 * torch.ones((10,), dtype=torch.int32)
# Reshape targets, so that cuDNN can be used
targets = torch.cat(tuple(targets[i, :target_lengths[i]] for i in range(10)))

# CPU: OK
print(torch.nn.functional.ctc_loss(logprobs, targets, input_lengths, target_lengths))

# CUDA, PyTorch native implementation: OK
torch.backends.cudnn.enabled = False
print(torch.nn.functional.ctc_loss(logprobs.to('cuda'), targets.to('cuda'), input_lengths, target_lengths))

# CUDA, cuDNN implementation: CRASHES (cuDNN expects targets in CPU).
torch.backends.cudnn.enabled = True
print(torch.nn.functional.ctc_loss(logprobs.to('cuda'), targets.to('cuda'), input_lengths, target_lengths))