import copy
import torch
import torch.ao.quantize_fx as quantize_fx

m = M(...)
mp = prepare_fx(copy.deepcopy(m), ...)
# wrong model used for calibration
calibrate(m, data_loader)
mq = convert_fx(mp)
evaluate(mq, data_loader)