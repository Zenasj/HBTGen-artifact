import torch

def fn(mean, std, tout):
    torch.normal(mean, std, out=tout)
    return tout

if __name__ == "__main__":
    tout = torch.empty([10]).to(torch.float)
    fn = torch.compile(fn)
    print(
        fn(
            torch.tensor(1).to(torch.float),
            torch.tensor(5).to(torch.float),
            tout,
        )
    )