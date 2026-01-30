import torch

from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
import traceback
import sys

class NanDetect(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args, kwargs=None):
        kwargs = kwargs or {}
        try:
            res = func(*args, **kwargs)
            flat_res, _ = tree_flatten(res)
            for t in flat_res:
                if not torch.is_tensor(t):
                    continue
                try:
                    if (t != t).any():
                        raise RuntimeError(f"Function {func}(*{args}, **{kwargs}) returned a NaN")
                except NotImplementedError:
                    pass
            return res
        except Exception as e:
            print(f"Error in NanDetect for function: {func}")
            print(f"Error message: {str(e)}")
            print("Stack trace:")
            traceback.print_exc()
            raise  # Re-raise the exception after printing the traceback