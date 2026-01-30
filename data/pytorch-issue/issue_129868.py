import gc
import os
import subprocess
import time

import torch.distributed as dist


if __name__ == "__main__":
    assert dist.is_available()

    dist.init_process_group()

    for i in range(10):
        grp = dist.new_group(backend="gloo")

        cmd = ["lsof", "-l", "-p", str(os.getpid())]
        out = subprocess.run(cmd, capture_output=True, text=True)

        print(
            f"Process {dist.get_rank()}, iteration {i + 1}:",
            len(out.stdout.split("\n")),
            "open FDs",
        )

        del grp
        gc.collect()

        # FDs are not released, even though group should have been torn down?

        time.sleep(0.1)