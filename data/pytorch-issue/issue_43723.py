import torch.nn as nn

import torch
import torch.nn.functional as F
print(torch.__version__)

weight_tensor = torch.tensor(
    [[0., 1., 2., 3.],
    [4., 5., 6., 7.],
    [8., 9., 10., 11.]]
)

weight_tensor_non_contig = weight_tensor[:, :3]  # This is non-contiguous strided.
print(f"weight_tensor_non_contig stride: {weight_tensor_non_contig.stride()}")

weight_tensor_contig = weight_tensor_non_contig.clone().contiguous()  # Contig-strided.
print(f"weight_tensor_contig stride: {weight_tensor_contig.stride()}")

index = torch.tensor([0, 1, 2])
offsets = torch.tensor([0, 2])
# Run embedding bag. What this should do is take segment mean, where the segments are
# defined by the offsets. In this case, 0:2 is one segment, 2:3 is one segment
output_non_contig = F.embedding_bag(
    input=index,
    weight=weight_tensor_non_contig,
    offsets=offsets,
    mode="mean",
)
output_contig = F.embedding_bag(
    input=index,
    weight=weight_tensor_contig,
    offsets=offsets,
    mode="mean",
)
print(f"Input Tensor:\n{iweight_tensor_contig}")

# The outputs returned in the non-contiguous version is wrong.
print(f"Non-contigous result (wrong):\n{output_non_contig}")

# The outputs returned in the contiguous version is correct.
print(f"Contigous result   (correct):\n{output_contig}")