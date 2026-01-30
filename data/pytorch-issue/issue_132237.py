import torch
from numpy import testing

# using fullgraph to raise an exception in place of graph breaks
@torch.compile(backend='inductor', fullgraph=True) 
def fn(x):
    if hasattr(x, "attr"):
        return 1
    else:
        return -1


def main():
    fn(testing)


if __name__ == "__main__":
    main()

hasattr