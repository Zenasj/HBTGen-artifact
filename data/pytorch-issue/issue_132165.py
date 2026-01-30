py
global_flag = False


def set_flag_true():
    global global_flag
    global_flag = True

def set_falg_false():
    global global_flag
    global_flag = False

py
import test_import
import torch


@torch.compile()
def fn(x):
    test_import.set_flag_true()
    test_import.set_falg_false()
    return x + 1

fn(torch.ones(2, 2))