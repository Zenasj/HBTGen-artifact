import torch
import torch.nn as nn

default_dynamic_qconfig = QConfigDynamic(activation=default_dynamic_quant_observer,
                                         weight=torch.nn.Identity)