# torch.rand(1, dtype=torch.int, device="cuda")  # Input is a single integer (e.g., iteration count)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.t_gpu_orig = torch.tensor(42, device="cuda")
        self.graph = None
        self.captured = False
        self.output = None  # Stores the cloned tensor from capture phase

    def forward(self, x):
        if not self.captured:
            # Capture CUDA graph during first call
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                # Clone the original tensor (part of captured operations)
                self.output = self.t_gpu_orig.clone()
            # Delete original tensor after capture to simulate memory reuse scenario
            del self.t_gpu_orig
            self.captured = True
            return self.output  # Return captured output
        else:
            # Replay the graph in subsequent calls
            self.graph.replay()
            return self.output  # Return output after replay

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random integer input (matches model's dummy iteration parameter)
    return torch.randint(0, 6, (1,), dtype=torch.int, device="cuda")

