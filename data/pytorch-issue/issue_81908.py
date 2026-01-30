import torch.nn as nn

from torch.testing._internal import common_utils  # Loading module in this file affects other test files
#  (...)
import torch

# PyTorch's `common_utils` is loaded to reference `TestCase`, forcing
#   execution of `torch.backends.disable_global_flags()` in global scope
class TestONNXRuntime(common_utils.TestCase):
    def test_detectron2_mask_rcnnfpn(self, batch_size=2):
        pass

import unittest
import torch
from detectron2.engine import DefaultTrainer, SimpleTrainer, default_setup, hooks
#  (...)
# `test_setup_config` fails even with `torch.backends.cudnn.flags()` context manager
#   as suggested by `RuntimeError: not allowed to set torch.backends.cudnn flags after disable_global_flags; please use flags() context manager instead` 
class TestTrainer(unittest.TestCase):
    #  (...)
    def test_setup_config(self):
        with tempfile.TemporaryDirectory(
            prefix="detectron2_test"
        ) as d, torch.backends.cudnn.flags():
            cfg = get_cfg()
            cfg.OUTPUT_DIR = os.path.join(d, "yacs")
            default_setup(cfg, {})  # <-- Failure is triggered here