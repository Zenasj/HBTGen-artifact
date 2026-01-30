import torch

ret_fut = torch.futures.Future()
def rpc_done_cb(rpc_fut):
    result = rpc_fut.wait() # throws error
    ret_fut.set_result(result) # will never be called

rpc_fut = rpc.rpc_async(...)
rpc_fut.add_done_callback(rpc_done_cb)
ret_fut.wait()