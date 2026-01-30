import torch.nn as nn

import gc

import torch
from torch.quantization import quantize_fx
from torch.quantization import fake_quantize


def is_tensor(x) -> bool:
    # https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741/3
    try:
        return (
            torch.is_tensor(x) or
            torch.is_tensor(getattr(x, "data", None))
        )
    except:
        return False


def live_tensors():
    # Only record id and repr, because we don't want to extend the lifetime
    # by holding a reference.
    return {
        id(obj): repr(obj) for obj in gc.get_objects()
        if is_tensor(obj)
    }


def print_new_tensors(initial_live_tensors):
    for tensor_id, tensor_repr in live_tensors().items():
        if tensor_id not in initial_live_tensors:
            print(tensor_repr)


def quantize_model():
    gc.collect()
    quantize_fx.prepare_qat_fx(
        torch.nn.ReLU(),
        {"": torch.quantization.get_default_qat_qconfig('fbgemm')},
    )
    gc.collect()


initial_live_tensors = live_tensors()
quantize_model()
print("Leaked Tensors:")
print_new_tensors(initial_live_tensors)

print("\nLeaked Tensors after _PartialWrapper args clear:")
from torch.quantization import observer
for obj in gc.get_objects():
    if isinstance(obj, observer._PartialWrapper):
        obj.callable_args.clear()
gc.collect()
print_new_tensors(initial_live_tensors)