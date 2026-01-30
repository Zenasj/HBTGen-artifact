import torch
import torch.nn as nn

class TestFullyShardUnshardMultiProcess(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 2)

    @skip_if_lt_x_gpu(2)
    def test_unshard_async(self):
        class ToyModel(nn.Module):
            def __init__(self, dim: int):
                super().__init__()
                self.linears = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.Linear(dim, dim, bias=False), nn.Linear(dim, dim, bias=False))
                self.proj = nn.Linear(dim, dim, bias=False)

            def forward(self, x: torch.Tensor):
                assert isinstance(self.linears[0].weight.data, DTensor), f"{type(self.linears[0].weight.data)=}"
                y = self.linears(x)
                y = self.proj(y)
                return y

        batch_size, dim = 2, 8
        torch.manual_seed(42)
        model = ToyModel(dim)
        fully_shard(model.linears[0])
        fully_shard(model.linears[1])
        fully_shard(model.linears[2])
        fully_shard(model.linears)
        replicate(model.cuda())
        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn((batch_size, dim), device="cuda")
        self.assertTrue(isinstance(model.linears[0].weight.data, DTensor))
        model(inp).sum()