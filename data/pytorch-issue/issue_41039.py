import torch

print(torch.__version__)

def nan_error() -> float:
    nan = float('nan')
    print(nan)
    return torch.tensor(nan)

if __name__ == "__main__":
    print('regular function:')
    regular_res = nan_error()
    print('Is nan?', torch.isnan(regular_res))

    print('script function:')
    scripted_f = torch.jit.script(nan_error)
    scripted_res = scripted_f()
    print('Is nan?', torch.isnan(scripted_res))