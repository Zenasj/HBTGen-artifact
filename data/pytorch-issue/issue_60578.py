import os
import torch
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc


if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '9867'
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    rpc.init_rpc("0",
                 rank=0,
                 world_size = 1,
                 backend=rpc.BackendType.TENSORPIPE,
                 rpc_backend_options=rpc.TensorPipeRpcBackendOptions(rpc_timeout=6000,
                                                                     init_method='env://')
                 )

    rpc.shutdown()