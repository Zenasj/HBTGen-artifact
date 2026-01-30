import torch
def fn():
    res = torch.tensor(data=[[1., -1.]])
    return res

if __name__ == "__main__":
    fn = torch.compile(fn)
    res_fwd = fn()
    print(res_fwd)