import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# using fullgraph to raise an exception in place of graph breaks
@torch._dynamo.config.patch(reorderable_logging_functions={logger.info})
@torch.compile(backend='eager', fullgraph=True) 
def fn(x):
    y = x + 1
    logger.info("hi")
    y = x * 2
    return y


def main():
    a = torch.tensor([1.,]).requires_grad_(True)
    fn(a)


if __name__ == "__main__":
    main()