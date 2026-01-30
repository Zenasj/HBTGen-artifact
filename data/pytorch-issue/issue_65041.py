import torch.nn as nn

import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

from torch.distributed.elastic.multiprocessing.errors import record

def parse_args(argv):
    parser = argparse.ArgumentParser(description="test script")
    parser.add_argument("--init_method", type=str, default="env://")
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--throw", action="store_true", default=False)
    parser.add_argument("--exit", action="store_true", default=False)
    return parser.parse_args()

record
def main():
    args = parse_args(sys.argv[1:])

    if args.throw:
        raise RuntimeError("rasing error since --throw was specified")

    if args.exit:
        sys.exit(1)

    init_method=args.init_method
    backend=args.backend

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    print(f"initializing `{backend}` process group with rank={rank}, world_size={world_size} at {init_method}")

    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank)

    print(f"successfully initialized process group with rank={dist.get_rank()}, world_size={dist.get_world_size()}")

    t = F.one_hot(torch.tensor(rank), num_classes=world_size)
    dist.all_reduce(t)
    derived_world_size = torch.sum(t).item()
    if derived_world_size != world_size:
        raise RuntimeError(f"derived world size: {derived_world_size} != actual world size: {world_size}")
    else:
        print(f"sucessfully derived world size: {derived_world_size} (expected: {world_size}). Exiting")

if __name__ == "__main__":
    main()

{"message": {"message": "SignalException: Process 17912 got signal: 15", "extraInfo": {"py_callstack": "Traceback (most recent call last):\n  File \"/gpfswork/rech/six/commun/conda/py38-pt111/lib/python3.8/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py\", line 345, in wrapper\n    return f(*args, **kwargs)\n  File \"/gpfswork/rech/six/commun/conda/py38-pt111/lib/python3.8/site-packages/torch/distributed/run.py\", line 724, in main\n    run(args)\n  File \"/gpfswork/rech/six/commun/conda/py38-pt111/lib/python3.8/site-packages/torch/distributed/run.py\", line 715, in run\n    elastic_launch(\n  File \"/gpfswork/rech/six/commun/conda/py38-pt111/lib/python3.8/site-packages/torch/distributed/launcher/api.py\", line 131, in __call__\n    return launch_agent(self._config, self._entrypoint, list(args))\n  File \"/gpfswork/rech/six/commun/conda/py38-pt111/lib/python3.8/site-packages/torch/distributed/launcher/api.py\", line 236, in launch_agent\n    result = agent.run()\n  File \"/gpfswork/rech/six/commun/conda/py38-pt111/lib/python3.8/site-packages/torch/distributed/elastic/metrics/api.py\", line 125, in wrapper\n    result = f(*args, **kwargs)\n  File \"/gpfswork/rech/six/commun/conda/py38-pt111/lib/python3.8/site-packages/torch/distributed/elastic/agent/server/api.py\", line 709, in run\n    result = self._invoke_run(role)\n  File \"/gpfswork/rech/six/commun/conda/py38-pt111/lib/python3.8/site-packages/torch/distributed/elastic/agent/server/api.py\", line 850, in _invoke_run\n    time.sleep(monitor_interval)\n  File \"/gpfswork/rech/six/commun/conda/py38-pt111/lib/python3.8/site-packages/torch/distributed/elastic/multiprocessing/api.py\", line 60, in _terminate_process_handler\n    raise SignalException(f\"Process {os.getpid()} got signal: {sigval}\", sigval=sigval)\ntorch.distributed.elastic.multiprocessing.api.SignalException: Process 17912 got signal: 15\n", "timestamp": "1645809651"}}}

from torch.distributed.elastic.multiprocessing.errors import record

@record
def main():
     pretrain(train_valid_test_datasets_provider, model_provider, forward_step,
              args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})