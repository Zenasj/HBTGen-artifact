import torch
from torchvision import models

def get_seq_exec_list(model):
    DUMMY_INPUT = torch.randn(1,3,224,224)
    model.eval()
    if (torch.cuda.is_available()):
        DUMMY_INPUT = DUMMY_INPUT.cuda()
    traced = torch.jit.trace(model, (DUMMY_INPUT,), check_trace=False)
    seq_exec_list = traced.code
    seq_exec_list = seq_exec_list.split('\n')[2:] # remove first two lines: (a. function name, and b. input_name)
    seq_exec_list = list(filter(None, seq_exec_list)) # remove empty strings
    for idx, item in enumerate(seq_exec_list):
        print("[{}]: {}".format(idx, item))

x = models.inception_v3()
get_seq_exec_list(x)

x = models.inception_v3(pretrained=True)
get_seq_exec_list(x)