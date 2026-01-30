import torch

def fn(mean, std, size, tout):
    torch.normal(mean, std, size, out=tout)
    return tout

if __name__ == "__main__":
    tout = torch.empty([10], dtype=torch.float32)
    fn = torch.compile(fn)
    print(
        fn(
            1.0,
            5.0,
            [10],
            tout,
        )
    )