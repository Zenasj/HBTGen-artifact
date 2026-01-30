3
import itertools as it
import torch as t
import torch.multiprocessing as mp

def infer(id, tensor):
    print(id)
    print(tensor)
    # del tensor immediately doesn't solve the problem
    del tensor

def main():
    # some global tensor
    g_tensor = t.full([1000, 1000], 2, device="cuda:0")
    g_tensor.share_memory_()

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(2)
    for i in range(10000000):
        print("start")
        pool.starmap(infer, zip(range(5), it.repeat(g_tensor)))

        # cpu tensors work just fine
        # for cuda tensors:
        # if I delete the global tensor, reassign it with a new cuda tensor
        # or if I use a tensor created dynamically in each iteration
        # the program freezes after 2 iterations.
        # Comment out the following lines and everything will work fine.
        del g_tensor
        g_tensor = t.full([1000, 1000], 2, device="cuda:0")
        g_tensor.share_memory_()

if __name__ == "__main__":
    main()