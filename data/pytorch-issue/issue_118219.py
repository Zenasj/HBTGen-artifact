import torch
import torch.nn as nn

def test_dynamo_dtensor_AAA(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # test passing in DTensor as inputs/outputs and run some tensor computation
        def fn(x, y, z):
            tmp = x.permute(0, 2, 1).contiguous()
            out = torch._C._nn.linear(tmp, y, z)
            return out.permute(0, 2, 1)

        x = DTensor.from_local(torch.randn(4, 32, 4, requires_grad=True), mesh, [Shard(0)], run_check=False)
        y = DTensor.from_local(torch.randn(4, 32, requires_grad=True), mesh, [Shard(0)], run_check=False)
        z = DTensor.from_local(torch.randn(4, requires_grad=True), mesh, [Shard(0)], run_check=False)
        ref = fn(x, y, z)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x, y, z)
        self.assertEqual(res, ref)