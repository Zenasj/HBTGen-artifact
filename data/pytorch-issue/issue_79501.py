# torch.rand(32, 1, 512, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(512, 512, 24)  # Matches the user's GRU configuration

    def forward(self, input):
        # Batched inference (method1)
        h0_batch = torch.zeros(24, 1, 512, dtype=input.dtype, device=input.device)
        output_batch, _ = self.gru(input, h0_batch)
        
        # Step-by-step inference (method2, replicates user's flawed h0 reset)
        output_step = []
        for t in range(input.size(0)):
            h0 = torch.zeros(24, 1, 512, dtype=input.dtype, device=input.device)
            step_input = input[t:t+1]
            out_step, _ = self.gru(step_input, h0)
            output_step.append(out_step)
        output_step = torch.cat(output_step, dim=0)
        
        # Compare outputs per frame using torch.allclose
        comparisons = []
        for t in range(input.size(0)):
            comp = torch.allclose(
                output_batch[t:t+1],
                output_step[t:t+1],
                rtol=1e-05,  # Default tolerances
                atol=1e-08
            )
            comparisons.append(comp)
        return torch.tensor(comparisons, device=input.device)

def my_model_function():
    return MyModel()  # Returns the fused model with comparison logic

def GetInput():
    return torch.rand(32, 1, 512, dtype=torch.float32)  # Matches input shape from the issue

