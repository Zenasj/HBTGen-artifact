import torch.nn as nn

import argparse
import multiprocessing
import multiprocessing.dummy
import os
import pickle
import queue
import random
import sys
import subprocess
import tempfile
import time

import torch
from torch.utils.benchmark import Timer, Compare


NUM_CORES = multiprocessing.cpu_count()
ENVS = (
    "ref",
    "fast_torch_fn_check"
)


MIN_RUN_TIME = 1
REPLICATES = 300
SETUP = """
x = torch.ones((1, 1))
y = torch.ones((1, 1))
"""

TASKS = {
    "tensor.py: _wrap_type_error_to_not_implemented `__floordiv__`": "x // y",
    "tensor.py: method          `__hash__`": "hash(x)",
    "functional.py: (unary)     `unique`": "torch.functional.unique(x)",
    "functional.py: (args)      `atleast_1d`": "torch.functional.atleast_1d((x, y))",
    "nn/functional.py: (unary)  `relu`": "torch.nn.functional.relu(x)",
    "nn/functional.py: (args)   `linear`": "torch.nn.functional.linear(x, y)",
}


def worker_main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str)
    output_file = parser.parse_args(argv).output_file

    env = os.path.split(os.getenv("CONDA_PREFIX"))[1]
    assert env in ENVS

    results = []
    for i, stmt in enumerate(TASKS.values()):
        timer = Timer(
            stmt=stmt,
            setup=SETUP,
            env=env,
            sub_label=" ",
            description=f"[{i}]",
        )
        results.append(timer.blocked_autorange(min_run_time=MIN_RUN_TIME))

    with open(output_file, "wb") as f:
        pickle.dump(results, f)


def main():
    num_workers = int(NUM_CORES // 2)
    tasks = list(ENVS * REPLICATES)
    random.shuffle(tasks)
    task_queue = queue.Queue()
    for t in tasks:
        task_queue.put(t)
    results = []

    def map_fn(worker_id):
        core = str(worker_id * 2)
        _, output_file = tempfile.mkstemp(suffix=".pkl")
        try:
            while True:
                try:
                    env = task_queue.get_nowait()
                except queue.Empty:
                    break

                subprocess.run(
                    " ".join([
                        "source", "activate", env, "&&",
                        "taskset", "--cpu-list", core,
                        "python", os.path.abspath(__file__),
                        "--mode", "worker",
                        "--output_file", output_file
                    ]),
                    shell=True,
                    check=True,
                )

                # We don't need a lock, as the GIL is enough.
                with open(output_file, "rb") as f:
                    results.extend(pickle.load(f))
        finally:
            os.remove(output_file)

    with multiprocessing.dummy.Pool(num_workers) as pool:
        st, eta, n_total = time.time(), "", len(tasks) * len(TASKS)
        map_job = pool.map_async(map_fn, range(num_workers))
        while not map_job.ready():
            n_complete = len(results)
            if n_complete:
                sec_per_element = (time.time() - st) / n_complete
                n_remaining = n_total - n_complete
                eta = f"ETA: {n_remaining * sec_per_element:.0f} sec"

            print(f"\r{n_complete} / {n_total}   {eta}".ljust(40), end="")
            sys.stdout.flush()
            time.sleep(2)
        print()

    compare = Compare(results)
    compare.trim_significant_figures()
    compare.colorize()
    compare.print()
    for i, k in enumerate(TASKS.keys()):
        print(f"[{i}] {k}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=("main", "worker"), default="main")
    args, remaining = parser.parse_known_args()

    if args.mode == "main":
        assert not remaining
        main()
    else:
        worker_main(remaining)