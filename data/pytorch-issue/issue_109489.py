import torch

# Put these 4 lines at the top of the main script, before any imports
# TODO: remove once https://github.com/pytorch/pytorch/issues/109489 is resolved
from torch._inductor import utils
utils._use_template_for_cuda = lambda x, y: True