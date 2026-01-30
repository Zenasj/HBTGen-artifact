import torch

# test_script.py
from torch.distributed.elastic.multiprocessing.errors import record

@record
def main():
    raise RuntimeError("foobar")

if __name__ == "__main__":
    main()