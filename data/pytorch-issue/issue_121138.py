# torch.rand(1, dtype=torch.float32)  # Dummy input shape to satisfy model requirements
import torch
import numpy as np

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pre-initialize problematic array to simulate HDF5 string data conversion scenario
        fill_value = "value".encode('ascii')
        self.problematic_data = np.full((400, 2000, 8), fill_value, dtype='|S6')  # Fixed-length ASCII strings
        
    def forward(self, dummy_input):
        """Attempt conversion that causes memory leak on exception"""
        try:
            # This will fail for ASCII string arrays, triggering exception handling
            return torch.from_numpy(self.problematic_data)
        except TypeError:
            return torch.tensor(-1)  # Return placeholder on failure
            
def my_model_function():
    return MyModel()

def GetInput():
    """Return dummy tensor to trigger model's conversion attempt"""
    return torch.rand(1)

