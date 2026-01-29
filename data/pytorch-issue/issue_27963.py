# Input is a PackedSequence containing tensors of shape (variable sequence length, 2)
import torch
from torch import nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
import copy

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.original = nn.Sequential(nn.LSTM(2, 5, num_layers=2))
        original_copy = copy.deepcopy(self.original)
        self.quantized = torch.quantization.quantize_dynamic(original_copy, dtype=torch.qint8)

    def forward(self, x):
        try:
            out1 = self.original(x)
            out2 = self.quantized(x)
        except:
            return torch.tensor([0], dtype=torch.bool)
        
        # Unpack outputs to compare
        output1, (h_n1, c_n1) = out1
        output2, (h_n2, c_n2) = out2
        
        # Pad the packed outputs to compare as tensors
        unpacked_out1, _ = pad_packed_sequence(output1)
        unpacked_out2, _ = pad_packed_sequence(output2)
        
        # Check if all parts are close within tolerance (quantization may introduce small diffs)
        if (torch.allclose(unpacked_out1, unpacked_out2, atol=1e-4) and
            torch.allclose(h_n1, h_n2, atol=1e-4) and
            torch.allclose(c_n1, c_n2, atol=1e-4)):
            return torch.tensor([1], dtype=torch.bool)
        else:
            return torch.tensor([0], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    sequences = [torch.rand(4, 2), torch.rand(3, 2), torch.rand(2, 2)]
    packed = pack_sequence(sequences, enforce_sorted=False)
    return packed

