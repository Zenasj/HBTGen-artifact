import ctypes as c
import torch

float_p    = c.POINTER(c.c_float)
longlong_p = c.POINTER(c.c_longlong)

class DLTensor(c.Structure):
    _fields_ = [
        ('data', float_p),
        ('device_type', c.c_int),
        ('device_id', c.c_int),
        ('ndim', c.c_int),
        ('code', c.c_uint8),
        ('bits', c.c_uint8),
        ('lanes', c.c_uint16),
        ('shape', longlong_p),
        ('strides', longlong_p),
        ('byte_offset', c.c_ulonglong),
        ('ctx', c.c_void_p),
        ('deleter', c.CFUNCTYPE(None, c.c_void_p))
    ]    

@c.CFUNCTYPE(None, c.c_void_p)
def deleter(tensor):
    print('Deleter called!')


def make_tensor(size):
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
    tensor.device_type = 1 # cpu
    tensor.device_id = 0
    tensor.ndim = 1
    tensor.code = 2 # float
    tensor.lanes = 1
    tensor.bits = 32
    tensor.shape = shape
    tensor.strides = strides
    tensor.ctx = 0
    tensor.deleter = deleter


    c.pythonapi.PyCapsule_New.argtypes = [c.c_void_p, c.c_char_p, c.c_void_p]
    c.pythonapi.PyCapsule_New.restype = c.py_object

    capsule = c.pythonapi.PyCapsule_New(c.cast(c.pointer(tensor), c.c_void_p), b'dltensor', None)

    a = torch.utils.dlpack.from_dlpack(capsule)
    print(a)


make_tensor(5)