import torch
# from transformers.utils import logging

# logger = logging.get_logger(__name__)

import logging
logger = logging.getLogger(__name__)


def func(x):
    # NOTE: Either of these lines will cause the export to fail.
    # logger.warning_once("abc")  # HF's logger
    # logger.log("abc")
    # print("abc")
    return x + 1


torch._dynamo.export(func, torch.randn(2, 3))
"""
torch._dynamo.exc.Unsupported: builtin: print [<class 'torch._dynamo.variables.constant.ConstantVariable'>] False

from user code:
   File "repro_logging.py", line 13, in func
    print("abc")

Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information

----------

torch._dynamo.exc.Unsupported: 'inline in skipfiles: Logger.log | log /lib/python3.10/logging/__init__.py, skipped according trace_rules.lookup'

from user code:
   File "repro_logging.py", line 13, in func
    logger.log("abc")

Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information
"""

import torch
from transformers.utils import logging as hf_logging

hf_logger = hf_logging.get_logger(__name__)

import logging
logger = logging.getLogger(__name__)


def func(x):
    print("abc")  # Pass with config patch
    # NOTE: Either of these lines will cause the export to fail.
    logger.log("abc")
    hf_logger.warning_once("abc")  # HF's logger
    return x + 1


with torch._dynamo.config.patch(reorderable_logging_functions={logging.log, hf_logging.warning_once, print}):
    torch._dynamo.export(func, torch.randn(2, 3))