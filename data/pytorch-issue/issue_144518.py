import torch

import logging
import warnings
torch._dynamo.config.reorderable_logging_functions = { warnings.warn, logging.warn, print }