cudart = ctypes.cdll.LoadLibrary(None)
res = cudart.cudaHostRegister(cptr, csize, cflags)