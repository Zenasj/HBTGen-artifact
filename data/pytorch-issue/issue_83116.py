import torch
import torch.fx

tracer = torch.fx.Tracer()

import copy
tracer_copy = copy.deepcopy(tracer)