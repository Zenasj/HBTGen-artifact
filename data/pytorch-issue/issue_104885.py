import torch

torch._logging.set_logs(recompiles=logging.DEBUG, dynamic=logging.INFO, guards=logging.INFO)