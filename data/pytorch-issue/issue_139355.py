import torch

@torch.no_grad()
def test():
    t_gpu_orig = torch.tensor(42, device="cuda")
    def func(i: int):
        # t_cpu = torch.tensor(i)
        t_gpu = t_gpu_orig.clone()
        return t_gpu  # t_cpu deconstructed
    graph = torch.cuda.CUDAGraph()
    for i in range(6):
        if i < 3:  # warmup
            y = func(i)
        elif i == 3:  # capture
            with torch.cuda.graph(graph):
                y = func(i)
            graph.replay()
            # Accidentally delete a referenced tensor.
            del t_gpu_orig
        else:  # replay
            t_gpu_new = torch.tensor(84, device="cuda")
            graph.replay()
        print(f"{i=}, {y.item()=}")

if __name__ == "__main__":
    test()