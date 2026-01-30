import torch._dynamo as dynamo
import torch.distributed as dist

@dynamo.optimize("eager")
def dist_available():
    return dist.is_available()

print(dist_available())