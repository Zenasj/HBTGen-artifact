import torch.nn as nn
import torch.nn.functional as F

import torch
from torch import nn

class SampleNet(nn.Module):
    def __init__(self):

        super().__init__()

        self.BatchNorm1 = nn.BatchNorm1d(5)

    def forward(self, X, X_len):
        # Mask
        X_pack = torch.nn.utils.rnn.pack_padded_sequence(X, X_len, batch_first=True, enforce_sorted=False)

        # norm
        X_ln = torch.nn.utils.rnn.PackedSequence(
            data=self.BatchNorm1(X_pack.data),
            batch_sizes=X_pack.batch_sizes,
            sorted_indices=X_pack.sorted_indices,
            unsorted_indices=X_pack.unsorted_indices,
        )
        
        return X_ln

example_input = torch.randn(1, 30, 5, requires_grad=True)
example_input_length = torch.abs(torch.randn(1, requires_grad=True)) + 1

print("Example input: %s" % example_input)
print("Example input lengths: %s" % list(example_input.size()))

net = SampleNet() # Model instantiation
net.eval()

example_output = net(example_input, example_input_length) # Check that forward pass works as expected:
print(f"Output from Net (works): {example_output}")


torch.onnx.export(model=net,
                  args=(example_input, example_input_length),
                  example_outputs=example_output,
                  f="test_pack_unpack.onnx",
                  verbose=True,
                  input_names=["x", "x_length"],
                  output_names=["preds"]
                 )# Export to ONNX (fails)

def my_pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True):

    lengths = torch.as_tensor(lengths, dtype=torch.int64)

    lengths, sorted_indices = torch.sort(lengths, descending=True)
    sorted_indices = sorted_indices.to(input.device)
    batch_dim = 0 if batch_first else 1
    input = input.index_select(batch_dim, sorted_indices)
    
    data, batch_sizes = torch._VF._pack_padded_sequence(input, lengths, batch_first)
  
    unsorted_indices = torch.empty_like(sorted_indices, memory_format=torch.legacy_contiguous_format)
    unsorted_indices.scatter_(0, sorted_indices, torch.arange(0, sorted_indices.numel(), device=sorted_indices.device))
    
    return data, batch_sizes, sorted_indices, unsorted_indices