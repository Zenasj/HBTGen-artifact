import torch.distributed as dist
import torch.distributed.rpc as rpc
import tempfile
import os.path

file = tempfile.NamedTemporaryFile(delete=False)
file_name = file.name
print(f"Using file: {file_name}, File Exists: {os.path.isfile(file_name)}")

# Init RPC using file
rpc_backend_options = rpc.TensorPipeRpcBackendOptions()
rpc_backend_options.init_method = f"file://{file_name}"
rpc.init_rpc("worker", rank=0, world_size=1, rpc_backend_options=rpc_backend_options)

# Init PG using file
dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{file_name}")

# Destroying PG also removes file
dist.destroy_process_group()
print(f"File Exists: {os.path.isfile(file_name)}")