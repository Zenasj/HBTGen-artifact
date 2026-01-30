import torch.nn as nn

import torch
import torch.nn.functional as F

def compute_cross_entropy(
    logits_test: torch.Tensor, labels_test: torch.Tensor
) -> torch.Tensor:
    return F.cross_entropy(
        logits_test.float(),
        labels_test,
        ignore_index=-100,
        reduction="sum",
    )


# Create dummy data
batch_size = 2
num_tokens = 16
vocab_size = 10
num_chunks = 8

logits = torch.randn(batch_size, num_tokens, vocab_size)
labels = torch.randint(0, vocab_size, (batch_size, num_tokens))

# Chunk and reshape logits
logits_chunks = [
    chunk.reshape(-1, chunk.size(-1)) for chunk in logits.chunk(num_chunks, dim=1)
]

# Chunk and reshape labels
labels_chunks = [chunk.reshape(-1) for chunk in labels.chunk(num_chunks, dim=1)]

# Avoid graph breaks
for l in labels_chunks:
    torch._dynamo.mark_dynamic(l, 0)

# Compile the compute_cross_entropy function
compiled_compute_cross_entropy = torch.compile(
    compute_cross_entropy, backend="inductor"
)

# Compute cross entropy for each chunk
total_loss = 0.0
for logits_chunk, labels_chunk in zip(logits_chunks, labels_chunks):
    total_loss += compiled_compute_cross_entropy(logits_chunk, labels_chunk)

print(total_loss)

# Avoid graph breaks
for l, logit in zip(labels_chunks, logits_chunks):
    torch._dynamo.mark_dynamic(l, 0)
    torch._dynamo.mark_dynamic(logit, 0)