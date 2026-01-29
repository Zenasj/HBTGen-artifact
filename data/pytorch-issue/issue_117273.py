import ctypes as c
import torch
from torch import nn

# torch.rand(B, 1, dtype=torch.int32)
class DLTensor(c.Structure):
    _fields_ = [
        ('data', c.POINTER(c.c_float)),
        ('device_type', c.c_int),
        ('device_id', c.c_int),
        ('ndim', c.c_int),
        ('code', c.c_uint8),
        ('bits', c.c_uint8),
        ('lanes', c.c_uint16),
        ('shape', c.POINTER(c.c_longlong)),
        ('strides', c.POINTER(c.c_longlong)),
        ('byte_offset', c.c_ulonglong),
        ('ctx', c.c_void_p),
        ('deleter', c.CFUNCTYPE(None, c.c_void_p))
    ]

@c.CFUNCTYPE(None, c.c_void_p)
def deleter(tensor):
    print('Deleter called!')

class MyModel(nn.Module):
    def forward(self, input_size):
        size = input_size.item()
        tensor = DLTensor()
        
        if size != 0:
            data = (c.c_float * size)()
            for i in range(size):
                data[i] = c.c_float(i)
        else:
            data = None

        shape = (c.c_longlong * 1)()
        shape[0] = size

        strides = (c.c_longlong * 1)()
        strides[0] = 1

        tensor.data = data
        tensor.device_type = 1  # CPU
        tensor.device_id = 0
        tensor.ndim = 1
        tensor.code = 2  # float
        tensor.lanes = 1
        tensor.bits = 32
        tensor.shape = shape
        tensor.strides = strides
        tensor.ctx = 0
        tensor.deleter = deleter  # Set the deleter

        capsule = c.pythonapi.PyCapsule_New(
            c.cast(c.pointer(tensor), c.c_void_p),
            b'dltensor',
            None
        )
        return torch.utils.dlpack.from_dlpack(capsule)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a tensor indicating the size (e.g., 0 to trigger the bug)
    return torch.tensor([0], dtype=torch.int32)

