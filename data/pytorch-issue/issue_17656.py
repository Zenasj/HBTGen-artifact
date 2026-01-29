# torch.rand(1, 10, 20, dtype=torch.float32, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        F = 20  # Feature dimension from original issue
        self.original_model = nn.LSTM(
            input_size=F,
            hidden_size=F,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        self.original_model.eval()
        self.original_model.to('cuda')  # Ensure model is on CUDA
        
        # Create sample input for tracing
        sample_input = torch.rand(1, 10, F, device='cuda')
        
        # Trace the original model
        traced_model = torch.jit.trace(self.original_model, sample_input)
        
        # Save and load the traced model to simulate the issue scenario
        traced_model.save('/tmp/test.pt')
        self.loaded_model = torch.jit.load('/tmp/test.pt', map_location='cuda')

    def forward(self, x):
        # Run both models and compare outputs
        original_output, _ = self.original_model(x)
        loaded_output, _ = self.loaded_model(x)
        
        # Return 1.0 if outputs are close, 0.0 otherwise (as a tensor)
        return torch.tensor(torch.allclose(original_output, loaded_output, atol=1e-5)).float()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 10, 20, dtype=torch.float32, device='cuda')

