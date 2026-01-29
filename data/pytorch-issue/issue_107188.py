import torch
import torch.nn as nn

# torch.rand(B, C, H, W, dtype=...) → Input shape is (1, 80) based on the decoder's input (80,80)
class MyModel(nn.Module):
    def forward(self, x):
        # Simulate hypotheses with scores derived from input tensor x
        b_hypos = [{'score': x[0, i]} for i in range(x.size(1))]  # List of dictionaries with scores
        # Extract scores into a Python list (problematic in Dynamo)
        scores_list = [hypo['score'] for hypo in b_hypos]
        # Convert list to tensor, which triggers the error when compiled
        scores_tensor = torch.tensor(scores_list)
        return scores_tensor

def my_model_function():
    return MyModel()

def GetInput():
    # Return a random tensor of shape (1, 80) to match the example in the original issue
    return torch.randn(1, 80)

