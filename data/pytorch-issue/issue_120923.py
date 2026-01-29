import torch
import transformers
from transformers import file_utils
from transformers.modeling_outputs import CausalLMOutputWithPast

# torch.rand(B=1, C=1, H=1, W=1, dtype=torch.float32)
class MyModel(torch.nn.Module):
    def forward(self, x):
        x = x + 1
        output = CausalLMOutputWithPast(loss=None, logits=x)
        return output[0]

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 1, 1, dtype=torch.float32)

