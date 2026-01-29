# torch.rand(1, dtype=torch.float32)
import torch
import torch.nn as nn
import threading
import contextvars

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        try:
            thread_result = torch.cat([x, FOO_threading.x])
        except:
            thread_result = None
        
        try:
            context_result = x + FOO_contextvar.get()
        except:
            context_result = None
        
        # Return 1 if both succeeded and outputs match, else 0
        if thread_result is None or context_result is None:
            return torch.tensor([0], dtype=torch.float32)
        else:
            return torch.tensor([1], dtype=torch.float32) if torch.allclose(thread_result, context_result) else torch.tensor([0], dtype=torch.float32)

def my_model_function():
    return MyModel()

def GetInput():
    # Initialize thread-local and contextvar state
    global FOO_threading
    FOO_threading = threading.local()
    FOO_threading.x = torch.zeros(1)
    
    global FOO_contextvar
    FOO_contextvar = contextvars.ContextVar("FOO_contextvar")
    FOO_contextvar.set(torch.zeros(1))
    
    return torch.ones(1, dtype=torch.float32)

