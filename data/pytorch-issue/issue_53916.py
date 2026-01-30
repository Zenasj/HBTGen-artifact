import torch

def checkIfNumericAnalyticAreClose(fail_test, analytical, numerical, rtol, atol, output_idx, input_idx, error_str=''):
    if not torch.allclose(analytical, numerical, rtol, atol):
        return fail_test(get_notallclose_msg(analytical, numerical, output_idx, input_idx,
                                            "Gradients failed to compare equal for grad output = 1j. "))
    return True