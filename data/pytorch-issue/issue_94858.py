import torch

log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

torch.utils.logging.set_verbosity(all=logging.INFO, dynamo=logging.DEBUG, graph=logging.ERROR)

torch.utils.logging.set_verbosity(log_level)