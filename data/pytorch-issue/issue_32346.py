import torch
from torch import nn
import torch.nn.functional as F

class ForwardWithDrop:
    def __init__(self, weights_names, module, dropout_p, original_forward):
        self.weights_names = weights_names
        self.module = module
        self.dropout_p = dropout_p
        self.original_forward = original_forward

    def __call__(self, *args, **kwargs):
        for name in self.weights_names:
            param = self.module._parameters.get(name)
            if param is None:
                raise RuntimeError(f"Parameter {name} not found in {self.module}")
            # Apply dropout to the parameter
            dropped_param = F.dropout(param, p=self.dropout_p, training=self.module.training)
            # Update the parameter in-place
            self.module._parameters[name] = nn.Parameter(dropped_param, requires_grad=param.requires_grad)
        return self.original_forward(*args, **kwargs)

def weight_drop(module, weights_names, dropout_p):
    original_forward = module.forward
    forward_wrapper = ForwardWithDrop(weights_names, module, dropout_p, original_forward)
    module.forward = forward_wrapper
    return module

# torch.rand(S, B, C=6, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.rnn = nn.LSTM(input_size=6, hidden_size=128, num_layers=1, batch_first=False)  # batch_first=False for (seq_len, batch, features)
        # Apply WeightDrop on 'weight_hh_l0'
        self.rnn = weight_drop(self.rnn, ['weight_hh_l0'], dropout_p=0.5)

    def forward(self, x):
        output, _ = self.rnn(x)
        return output

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random input tensor of shape (seq_len, batch_size, input_size)
    return torch.rand(10, 5, 6, dtype=torch.float32)

