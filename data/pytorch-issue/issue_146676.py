# torch.rand(100, 100, dtype=torch.int64)
import torch
import inspect
import warnings
from torch import nn

def verify_variable_names(func, local_vars, global_vars, check_vars=None):
    sig = inspect.signature(func)
    local_names = list(sig.parameters.keys())
    local_values = [local_vars[name] for name in local_names]
    external_var_names = {}
    for local_name, local_value in zip(local_names, local_values):
        if check_vars is not None and local_name not in check_vars:
            continue
        for global_name, global_value in global_vars.items():
            if id(global_value) == id(local_value):
                external_var_names[local_name] = global_name
                break
        if local_name not in external_var_names:
            warnings.warn(f"{local_name} in {func.__name__} not found as a valid variable in the global scope.")
    # print(f"external_var_names: {external_var_names}")

def checked_randint(low, high, size, out=None, dtype=None, layout=None, device=None, pin_memory=False, requires_grad=False):
    verify_variable_names(globals()[inspect.currentframe().f_code.co_name], locals(), globals(), check_vars=['out'])
    return torch.randint(low, high, size, out=out, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory, requires_grad=requires_grad)

class MyModel(nn.Module):
    def forward(self, out_tensor):
        # Original implementation (no check)
        orig_out = torch.randint(0, 10, (100, 100), out=out_tensor)
        # Checked implementation (with existence check)
        checked_out = checked_randint(0, 10, (100, 100), out=out_tensor)
        return orig_out, checked_out  # Return both outputs for comparison

def my_model_function():
    return MyModel()

def GetInput():
    global my_out
    my_out = torch.empty(100, 100, dtype=torch.int64)
    return my_out

