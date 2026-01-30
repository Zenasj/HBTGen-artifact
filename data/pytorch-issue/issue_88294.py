from torch.distributed import rpc
import os
import torch

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5679'
rpc.init_rpc("master", rank=0, world_size=1)

def func_return_arrays(arrays):
    return arrays
source_arr = torch.tensor([1.+0.1j, 2.-0.2j, 3.+0.3j, 4.-0.4j], dtype=torch.complex64, requires_grad = True)
ele_conj_arr = source_arr.clone()
for i in range(len(ele_conj_arr)):
    ele_conj_arr[i] = ele_conj_arr[i].conj()
conj_arr = source_arr.conj()
print("Directly use conj() function:  ")
print("Original tensor:  ", conj_arr)
rref = rpc.remote("master", func_return_arrays, args=(conj_arr,)) # Note: the bug is here, when transferring conjugate tensor, the conjugate information will be lost!
rpc_conj_arr = rref.to_here()
print("After RPC transfering tensor:  ", rpc_conj_arr)
print("Difference:  ", conj_arr-rpc_conj_arr)
print("  ")

print("Use conj() function elementwise:  ")
print("Original tensor:  ", ele_conj_arr)
rref = rpc.remote("master", func_return_arrays, args=(ele_conj_arr,))
rpc_ele_conj_arr = rref.to_here()
print("After RPC transferring tensor:  ", rpc_ele_conj_arr)
print("Difference:  ", ele_conj_arr-rpc_ele_conj_arr)