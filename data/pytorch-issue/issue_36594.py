#!/usr/bin/env python

import multiprocessing as mp

def worker(ipc_ptr, shape, dtype):
    import pycuda.driver as cuda
    import pycuda.autoinit
    import cupy
    from torch.utils import dlpack
    import torch
    # dummy_init = torch.ones(1, device=torch.device("cuda")) # uncomment to make it work as expected
    data_ptr = cuda.IPCMemoryHandle(ipc_ptr)
    buff = {'data': (int(data_ptr), False), 'shape': shape, 'typestr': dtype}
    class capsule(object):
        pass
    cap = capsule()
    cap.__cuda_array_interface__ = buff
    cuarray = cupy.asanyarray(cap)
    print("cupy array from IPC", cuarray.shape)
    pack = cuarray.toDlpack()
    cuarray2 = cupy.fromDlpack(pack)
    print("cupy array from DLPack", cuarray2.shape)
    pack2 = cuarray2.toDlpack()
    tensor = dlpack.from_dlpack(pack2)
    print("Pytorch Tensor from_dlpack", tensor.shape)

def pytorch_ipc_dlpack():
    import pycuda.driver as cuda
    import pycuda.autoinit
    import cupy
    tensor = cupy.ones([1, 2, 3])
    ipc_ptr = cuda.mem_get_ipc_handle(tensor.data.ptr)
    shape = tensor.shape
    dtype = str(tensor.dtype)
    proc = mp.Process(target=worker, args=(ipc_ptr, shape, dtype))
    proc.start()
    proc.join()

if __name__ == "__main__":
    mp.set_start_method("spawn")
    pytorch_ipc_dlpack()