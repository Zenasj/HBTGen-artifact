import tempfile
import torch.distributed.rpc as rpc

init_file = tempfile.mkstemp()[1]
options = rpc.TensorPipeRpcBackendOptions(init_method="file://" + init_file, _transports=["mpt_uv", "basic", "cuda_ipc", "cuda_gdr", "cuda_xth", "cuda_basic"])
rpc.init_rpc("worker0", rank=0, world_size=1, backend=rpc.BackendType.TENSORPIPE, rpc_backend_options=options)