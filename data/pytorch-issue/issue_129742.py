import torch

# using fullgraph to raise an exception in place of graph breaks
@torch.compile(backend='inductor', fullgraph=True) 
def fn(x):
    if hasattr(x, "attr"):
        return x + 1
    else:
        return x - 1


def main():
    t1 = torch.tensor([6.])
    t1.attr = False
    fn(t1)


if __name__ == "__main__":
    main()