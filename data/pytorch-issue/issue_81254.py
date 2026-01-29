# torch.randint(0, 5, (4,), dtype=torch.int64)  # Parameters: base_event_dim, domain_event_dim, codomain_event_dim, reinterpreted_batch_ndims
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, params):
        base_event_dim = params[0]
        domain_event_dim = params[1]
        codomain_event_dim = params[2]
        reinterpreted_batch_ndims = params[3]

        # Original formula: transform.codomain.event_dim + max(domain_event_dim - base_event_dim, 0)
        original_diff = domain_event_dim - base_event_dim
        original_max = torch.where(original_diff > 0, original_diff, torch.tensor(0, dtype=params.dtype))
        original_event_dim = codomain_event_dim + original_max

        # User's suggested formula: codomain_event_dim + max(reinterpreted_batch_ndims, 0)
        user_max = torch.where(reinterpreted_batch_ndims > 0, reinterpreted_batch_ndims, torch.tensor(0, dtype=params.dtype))
        user_suggestion = codomain_event_dim + user_max

        # Return comparison result as a boolean tensor
        return original_event_dim == user_suggestion

def my_model_function():
    return MyModel()

def GetInput():
    # Generate valid parameters:
    # base_event_dim, domain_event_dim, codomain_event_dim >=0; reinterpreted_batch_ndims can be negative
    base = torch.randint(0, 5, (1,)).item()
    domain = torch.randint(0, 5, (1,)).item()
    codomain = torch.randint(0, 5, (1,)).item()
    rebn = torch.randint(-2, 3, (1,)).item()  # -2 to 2 inclusive
    return torch.tensor([base, domain, codomain, rebn], dtype=torch.int64)

