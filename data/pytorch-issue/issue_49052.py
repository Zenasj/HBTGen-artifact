import torch.distributed as dist
import datetime

dist.TCPStore("127.0.0.1", 0, True, timedelta(seconds=30))