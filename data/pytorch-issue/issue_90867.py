import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.autograd import detect_anomaly

class MyMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args, kwargs=None):
        node = torch._C._current_autograd_node()
        print(f"Running {func} from within {node}")
        if node is not None:
            print("The Node was created at:")
            print("\n  ".join(node.metadata["traceback_"]))
        return func(*args, **kwargs or {})


with MyMode(), detect_anomaly():
    print("FW")
    a = torch.rand(10, requires_grad=True) 
    b = a.mul(2)
    b = b.div(3)
    b = b.sum()
    print("BW")
    b.backward()