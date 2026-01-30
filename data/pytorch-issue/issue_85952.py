import torch

import logging

# from functorch.compile import memory_efficient_fusion

_logger = logging.getLogger('aaaaa')

logging.root.addHandler(logging.StreamHandler())
logging.root.setLevel(logging.INFO)
    
_logger.info('some log info')