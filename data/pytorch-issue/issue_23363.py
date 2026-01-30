# node 0: 10.30.112.212
init_method="tcp://10.30.112.212:10087"
rank=0
world_size=2
backend="nccl"

# node 1: 10.30.112.224
init_method="tcp://10.30.112.212:10087"
rank=1
world_size=2
backend="nccl"

# node 0: 10.30.112.212
init_method="tcp://10.30.112.212:10087"
rank=0
world_size=2
backend="nccl"

# node 1: 10.30.112.212
init_method="tcp://10.30.112.212:10087"
rank=1
world_size=2
backend="nccl"