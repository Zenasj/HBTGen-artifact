# torch.rand(seq_len, batch_size, 29, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bigru = nn.GRU(input_size=29, hidden_size=29, num_layers=3, bidirectional=True, bias=True)
    
    def forward(self, x):
        # Save original reverse weights for layers 0 and 1
        saved_weights = []
        for i in range(2):  # layers 0 and 1 (since num_layers=3)
            ih = getattr(self.bigru, f'weight_ih_l{i}_reverse').data
            hh = getattr(self.bigru, f'weight_hh_l{i}_reverse').data
            saved_weights.append( (ih.clone(), hh.clone()) )
        
        # Compute prefill output
        out_prefill, _ = self.bigru(x)
        
        # Modify reverse weights to 100 for layers 0 and 1
        for i in range(2):
            layer = i
            getattr(self.bigru, f'weight_ih_l{layer}_reverse').data.fill_(100.)
            getattr(self.bigru, f'weight_hh_l{layer}_reverse').data.fill_(100.)
        
        # Compute postfill output
        out_postfill, _ = self.bigru(x)
        
        # Restore original weights
        for i in range(2):
            ih_orig, hh_orig = saved_weights[i]
            getattr(self.bigru, f'weight_ih_l{i}_reverse').data.copy_(ih_orig)
            getattr(self.bigru, f'weight_hh_l{i}_reverse').data.copy_(hh_orig)
        
        # Reshape outputs to separate directions
        out_prefill = out_prefill.view(x.size(0), x.size(1), 2, 29)
        out_postfill = out_postfill.view(x.size(0), x.size(1), 2, 29)
        
        # Calculate difference for forward direction (index 0)
        diff = (out_prefill[:, :, 0, :] - out_postfill[:, :, 0, :]).abs().mean()
        
        return diff > 1e-5  # returns a boolean indicating significant difference

def my_model_function():
    return MyModel()

def GetInput():
    seq_len = 5
    batch_size = 7
    dim = 29
    return torch.rand(seq_len, batch_size, dim, dtype=torch.float32)

