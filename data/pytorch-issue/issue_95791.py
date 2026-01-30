import torch
from torch_cluster import radius
import torch._dynamo as dynamo

def myradius(x: torch.Tensor, y: torch.Tensor, r: float,
             batch_x: Optional[torch.Tensor] = None,
             batch_y: Optional[torch.Tensor] = None,
             max_num_neighbors: int = 32,
             num_workers: int = 1) -> torch.Tensor:
    return radius(x,y,r,batch_x, batch_y, max_num_neighbors, num_workers)
device = torch.device('cuda:0')
x2 = torch.tensor([0.0], device=device)
y2 = torch.tensor([1.0], device=device)
dynamo.explain(myradius, x2, y2, 2)

def radius(x: torch.Tensor, y: torch.Tensor, r: float,
           batch_x: Optional[torch.Tensor] = None,
           batch_y: Optional[torch.Tensor] = None, max_num_neighbors: int = 32,
           num_workers: int = 1) -> torch.Tensor:

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y
    x, y = x.contiguous(), y.contiguous()

    batch_size = 1
    if batch_x is not None:
        assert x.size(0) == batch_x.numel()
        batch_size = int(batch_x.max()) + 1
    if batch_y is not None:
        assert y.size(0) == batch_y.numel()
        batch_size = max(batch_size, int(batch_y.max()) + 1)

    ptr_x: Optional[torch.Tensor] = None
    ptr_y: Optional[torch.Tensor] = None
    if batch_size > 1:
        assert batch_x is not None
        assert batch_y is not None
        arange = torch.arange(batch_size + 1, device=x.device)
        ptr_x = torch.bucketize(arange, batch_x)
        ptr_y = torch.bucketize(arange, batch_y)

    return torch.ops.torch_cluster.radius(x, y, ptr_x, ptr_y, r,
                                          max_num_neighbors, num_workers)

x = torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]], device=device).float()
batch_x = torch.tensor([0, 0, 0, 0], device=device)
y = torch.tensor([[-1, 0], [1, 0]], device=device).float()
batch_y = torch.tensor([0, 0], device=device)
dynamo.explain(myradius, x, y, 1.5, batch_x, batch_y)