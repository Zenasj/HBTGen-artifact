import torch

def fn(op_args, op_kargs):
    output = torch.normal(*op_args, **op_kargs)
    return output

if __name__ == "__main__":
    op_args = [2.0, 2.0, [2, 2, 2]]
    op_kargs = {'dtype': torch.bfloat16, 'layout': torch.strided}
    coml_fn = torch.compile(fn)
    res = coml_fn(op_args, op_kargs)
    print(res)