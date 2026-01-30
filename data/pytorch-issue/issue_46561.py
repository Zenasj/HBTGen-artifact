import torch.distributed.distributed_c10d as c10d
import datetime

def print_global_vars():
    print("===========")
    print("_pg_map", c10d._pg_map)
    print("_pg_names", c10d._pg_names)
    print("_pg_group_ranks", c10d._pg_group_ranks)
    print("_default_pg", c10d._default_pg)
    print("_default_pg_init_method", c10d._default_pg_init_method)
    print("_group_count", c10d._group_count)

print_global_vars()
try:
    c10d.init_process_group("gloo", timeout=datetime.timedelta(milliseconds=1), rank=0, world_size=2, init_method=f"tcp://127.0.0.1:8000")
except RuntimeError:
    pass

print_global_vars()

while True:
    try:
        torch.distributed.init_process_group(...)
        break
    except Exception:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()