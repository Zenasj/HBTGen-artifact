# torch.rand(3, 4, device='cuda'), torch.tensor([2, 1, 4], device='cpu')
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, inputs):
        batch_input, seq_length = inputs
        # Pack with CPU lengths (allowed in all versions now)
        packed_cpu = torch.nn.utils.rnn.pack_padded_sequence(
            batch_input, seq_length.to('cpu'), batch_first=True, enforce_sorted=False
        )
        unpacked_cpu, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_cpu, batch_first=True)
        
        # Try packing with GPU lengths (allowed in older versions)
        try:
            packed_gpu = torch.nn.utils.rnn.pack_padded_sequence(
                batch_input, seq_length.to(batch_input.device), batch_first=True, enforce_sorted=False
            )
            unpacked_gpu, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_gpu, batch_first=True)
            success_gpu = True
        except RuntimeError:
            unpacked_gpu = None
            success_gpu = False
        
        if success_gpu:
            # Compare the two unpacked tensors
            result = torch.allclose(unpacked_cpu, unpacked_gpu)
        else:
            # If GPU approach failed, the discrepancy exists
            result = False
        
        return torch.tensor([result], dtype=torch.float32)

def my_model_function():
    return MyModel()

def GetInput():
    gpu_device = torch.device("cuda:0")
    batch_input = torch.rand(3, 4, device=gpu_device)  # Using random for testing
    seq_length = torch.tensor([2, 1, 4], device='cpu')
    return (batch_input, seq_length)

