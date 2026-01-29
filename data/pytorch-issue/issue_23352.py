# torch.rand(B, C, H, W, dtype=...)  # Add a comment line at the top with the inferred input shape
import torch
from torch import nn
from torch.nn import Parameter

INFER = object()

class UninitializedParameter(Parameter):
    def __new__(cls, infer_size_fn, dtype=torch.float, requires_grad=True):
        dummy_data = torch.tensor([], dtype=dtype)
        return nn.Parameter.__new__(cls, dummy_data, requires_grad)
    
    def __init__(self, infer_size_fn, dtype=torch.long):
        self.infer_size_fn = infer_size_fn
        
    def finalize(self, *args, **kwargs):
        requires_grad, self.requires_grad = self.requires_grad, False
        self.resize_(*self.infer_size_fn(*args, **kwargs))
        delattr(self, "infer_size_fn")
        self.requires_grad = requires_grad

class ModuleWithUninitialized(nn.Module):
    def __init__(self):
        super().__init__()
        self._finalize_handle = self.register_forward_pre_hook(self._finalize)
        
    def _finalize(self, *input):
        for key, parameter in self._parameters.items():
            if isinstance(parameter, UninitializedParameter):
                parameter.finalize(*input)
                self._parameters[key] = Parameter(parameter.data, parameter.requires_grad)
        if hasattr(self, "reset_parameters"):
            self.reset_parameters()
        self._finalize_handle.remove()
        delattr(self, "_finalize_handle")

class LazyLinear(nn.Linear, ModuleWithUninitialized):
    def __init__(self, in_features, out_features, bias=True):
        ModuleWithUninitialized.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        if in_features is INFER:
            self.weight = UninitializedParameter(self._resolve_inferred_input)
        else:
            self.weight = Parameter(torch.Tensor(out_features, in_features))
            self.reset_parameters()
            
    @staticmethod
    def _resolve_inferred_input(self, inputs):
        (input,) = inputs
        self.in_features = input.size()[-1]
        return [self.out_features, self.in_features]

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(100, 10)  # Example embedding
        self.representation = LazyLinear(INFER, 50)  # Lazy Linear layer
        self.decoder = nn.Linear(50, 10)  # Decoder layer

    def forward(self, x):
        embedded = self.embedding(x)
        representation = self.representation(embedded)
        return self.decoder(representation)

def my_model_function():
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randint(0, 100, (8, 15))  # Example input

