import torch
import torch.nn as nn

class BatchRNN(nn.Module):

  def __init__(self):
    super().__init__()

  def forward(self, inputs, input_paddings):
    batch_size, seq_len, _ = inputs.size()
    lengths = torch.randint(1, seq_len + 1, (batch_size,))
    packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(
        inputs, lengths, batch_first=True, enforce_sorted=False)

    return packed_inputs

model = BatchRNN()
model(torch.randn(1, 32, 32), torch.randn(1, 32, 32))
model = torch.compile(model) 
model(torch.randn(1, 32, 32), torch.randn(1, 32, 32))

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
import torch.compiler

def custom_pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False):
    if not batch_first:
        # Inputs should have shape (batch_size, seq_len, *)
        inputs = inputs.transpose(0, 1)

    if not enforce_sorted:
        lengths, sorted_indices = lengths.sort(descending=True)
        inputs = inputs[sorted_indices]

    # Create a list to hold packed values
    packed_data = []
    batch_sizes = []

    # Iterate over each time step and store non-padded values
    for i in range(lengths.max()):
        batch_size_at_i = (lengths > i).sum().item()
        batch_sizes.append(batch_size_at_i)
        packed_data.append(inputs[:batch_size_at_i, i])

    packed_data = torch.cat(packed_data, dim=0)

    # Create a PackedSequence object to match the output of the original pack_padded_sequence
    packed_sequence = PackedSequence(
        data=packed_data,
        batch_sizes=torch.tensor(batch_sizes),
        sorted_indices=sorted_indices,
        unsorted_indices=sorted_indices.sort()[1]  # Get the indices to unsort the tensor
    )

    return packed_sequence

class BatchRNN(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, inputs, input_paddings):
        batch_size, seq_len, _ = inputs.size()
        lengths = torch.randint(1, seq_len + 1, (batch_size,))
        packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(
            inputs, lengths, batch_first=True, enforce_sorted=False)

        return packed_inputs

class BatchRNNCustom(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, inputs, input_paddings):
        batch_size, seq_len, _ = inputs.size()
        lengths = torch.randint(1, seq_len + 1, (batch_size,))
        packed_inputs = custom_pack_padded_sequence(
            inputs, lengths, batch_first=True, enforce_sorted=False)

        return packed_inputs

# Generate common inputs for both models
common_input = torch.randn(1, 2, 2)
common_lengths = torch.randint(1, 3, (1,))

# The original model works in eager
model = BatchRNN()
packed_output_1 = model(common_input, common_input)
torch.compiler.reset()

# The custom model works in eager
model_custom = BatchRNNCustom()
packed_output_custom = model_custom(common_input, common_input)

# Uncomment the following lines if you want to test the models in the compiled mode
# model_compiled = torch.compile(model) 
# packed_output_2 = model_compiled(torch.randn(1, 2, 2), torch.randn(1, 2, 2))
model_custom_compiled = torch.compile(model_custom)
custom_output = model_custom_compiled(common_input, common_input)

print(packed_output_1)
print(custom_output)