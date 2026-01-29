# torch.rand(1, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        with torch.profiler.profile() as p:
            x = x + 1  # Simple operation to profile
        # Trigger the error by accessing kineto_results on p.profiler
        p.profiler.kineto_results.experimental_event_tree()
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1)  # Matches the input shape (1,) in the example

