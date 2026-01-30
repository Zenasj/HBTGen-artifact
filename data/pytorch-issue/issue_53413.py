import torch.nn as nn

from torch.distributed import rpc
import torch
import tempfile
import faulthandler
tmpfile = tempfile.NamedTemporaryFile()
rpc.init_rpc(
    name="worker",
    rank=0,
    world_size=1,
    rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
        init_method="file://{}".format(tmpfile.name),
    )
)

from torch.distributed.pipeline.sync import Pipe

pipe1 = Pipe(torch.nn.Sequential(torch.nn.Linear(10, 10).cuda(0)))
pipe2 = Pipe(torch.nn.Sequential(torch.nn.Linear(10, 10).cuda(1)))

t = torch.rand(10, 10).cuda(0)

out = pipe2(pipe1(t).local_value().to(1)).local_value()
out.sum().backward()
faulthandler.dump_traceback_later(10)